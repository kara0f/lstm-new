# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:25:22 2025

@author: kara
"""

import numpy as np
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, Add
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D
import wfdb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import scipy.signal
import glob
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
# ====================== 数据预处理模块 ======================
def load_mit_bih_af_records(data_dir):
    """加载MIT-BIH AF数据库所有记录"""
    dat_files = glob.glob(os.path.join(data_dir, '*.dat'))
    "加载的dat文件中一般会有标注"
    records = [os.path.basename(f).split('.')[0] for f in dat_files]
    "自定义一个records文件，存储所有标注"
    return sorted(set(records))
"返回整理好的已标注的记录"

def preprocess_signal(signal, fs=200):
    """ECG信号预处理"""
    # 中值滤波去除基线漂移
    window_size = int(0.2 * fs)  # 200ms窗口
    if window_size % 2 == 0:
        window_size += 1
    baseline = scipy.signal.medfilt(signal, kernel_size=window_size)
    signal = signal - baseline
    
    # 三阶带通滤波 (5-17Hz)
    b, a = scipy.signal.butter(3, [5, 17], btype='bandpass', fs=fs)
    signal = scipy.signal.filtfilt(b, a, signal)
    
    # 标准化
    return (signal - np.mean(signal)) / np.std(signal)

def detect_r_peaks(signal, fs=200):
    """基于梯度的R峰检测算法"""
    # 计算一阶差分
    diff = np.diff(signal, prepend=signal[0])
    
    # 计算梯度幅度
    grad_mag = np.abs(diff)
    
    # 自适应阈值
    threshold = 0.5 * np.max(grad_mag)
    
    # 寻找候选峰值
    candidates = np.where(grad_mag > threshold)[0]
    
    # 合并邻近峰值
    peaks = []
    last_peak = -100
    for i in candidates:
        if i - last_peak > fs * 0.2:  # 200ms不应期
            # 在候选点附近寻找局部最大值
            search_window = signal[max(0, i-10):min(len(signal), i+10)]
            true_peak = np.argmax(search_window) + max(0, i-10)
            peaks.append(true_peak)
            last_peak = true_peak
    
    return np.array(peaks)

# ... 保留前面的导入和函数定义 ...

def extract_features(record, data_dir):
    """从单条记录中提取特征和标签 - 使用WFDB直接解析注释"""
    try:
        # 读取信号和注释
        signals, fields = wfdb.rdsamp(os.path.join(data_dir, record))
        annotation = wfdb.rdann(os.path.join(data_dir, record), 'atr')
        
        # 使用导联II (索引1)
        ecg = signals[:, 1]
        ecg = preprocess_signal(ecg, fs=fields['fs'])
        
        # 检测R峰
        r_peaks = detect_r_peaks(ecg, fs=fields['fs'])
        
        # 计算RR间期 (ms)
        rr_intervals = np.diff(r_peaks) / fields['fs'] * 1000
        
        # 创建标签向量 (0=正常, 1=AF)
        rr_labels = np.zeros(len(rr_intervals), dtype=int)
        
        
        # 获取所有心律注释点
        rhythm_changes = []
        for i in range(len(annotation.sample)):
            symbol = annotation.symbol[i]
            if symbol in ['N', 'V', 'A', ]:
                rhythm_changes.append({
                    'sample': annotation.sample[i],
                    'rhythm': symbol
                })
        
        if not rhythm_changes:
            print(f"  警告: 记录 {record} 没有检测到心律注释")
            return rr_intervals, rr_labels
        
        # 对每个RR间期进行标记
        current_rhythm = None
        next_change_index = 0
        
        # 按时间顺序处理每个RR间期
        for i in range(len(rr_intervals)):
            # 当前RR间期的起始和结束R峰位置
            start_r = r_peaks[i]
            end_r = r_peaks[i+1]
            
            # 检查是否需要更新当前心律
            while (next_change_index < len(rhythm_changes) and 
                   rhythm_changes[next_change_index]['sample'] <= end_r):
                current_rhythm = rhythm_changes[next_change_index]['rhythm']
                next_change_index += 1
            
            # 标记AF
            if current_rhythm in ['A', 'V']:
                rr_labels[i] = 1
        
        # 统计AF比例
        af_ratio = np.mean(rr_labels)
        print(f"  检测到 {np.sum(rr_labels)} 个AF事件, AF比例={af_ratio:.4f}")
        
        return rr_intervals, rr_labels
    
    except Exception as e:
        print(f"处理记录 {record} 时出错: {str(e)}")
        return None, None

# ... 保留其余代码不变 ...

def create_sequences(data, labels, window_size=30):
    """创建滑动窗口序列 - 针对AF检测优化"""
    if len(data) < window_size:
        return np.array([]), np.array([])
    
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        
        # 使用窗口内是否有AF事件作为标签
        # 如果窗口内有至少一个AF事件，则标记为AF
        if np.any(labels[i:i+window_size]):
            y.append(1)
        else:
            y.append(0)
    
    return np.array(X), np.array(y)

class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        # —— 一定要在 __init__ 里初始化 f1_scores
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        # 预测
        y_pred_prob = self.model.predict(X_val, batch_size=256, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        # 计算 F1
        f1 = f1_score(y_val, y_pred, average='binary')
        self.f1_scores.append(f1)
        print(f" — val_f1: {f1:.4f}")
        logs['val_f1'] = f1

# ====================== 模型构建模块 ======================
class AttentionLayer(tf.keras.layers.Layer):
    """自定义Attention层"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                               shape=(input_shape[-1], 1),
                               initializer='glorot_normal')
        self.b = self.add_weight(name='att_bias',
                               shape=(input_shape[1], 1),
                               initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # 计算注意力分数
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        # 计算注意力权重
        alpha = tf.nn.softmax(e, axis=1)
        # 加权求和
        context = tf.reduce_sum(x * alpha, axis=1)
        return context


def build_bilstm_attention_model(input_shape=(30, 1), num_classes=2):
    """构建一个 Conv1D + 双层 BiLSTM + Attention + 全连接 的网络"""
    inputs = Input(shape=input_shape)  # (batch, 30, 1)

    # # —— 1. 卷积层提取局部特征 —— 
    conv1 = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    bn1 = BatchNormalization()(conv1)
    
    # 残差捷径分支
    shortcut = Conv1D(64, 1, padding='same')(inputs)  # 1x1卷积匹配维度
    shortcut = BatchNormalization()(shortcut)
    
    # 残差相加
    res_out = Add()([bn1, shortcut])  # 使用导入的Add层
    pool1 = MaxPool1D(2)(res_out)
    # —— 2. 双层 BiLSTM —— 
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(pool1)  # -> (batch, 7, 128)
    
    x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.25))(x)  # -> (batch, 7, 64)
    
    # —— 3. Attention 层 —— 
    x = LayerNormalization()(x)
    att = AttentionLayer()(x)  # -> (batch, 64)

    # —— 4. 全连接 + Dropout + L2 正则 —— 
    fc = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(att)
    fc = Dropout(0.5)(fc)
    outputs = Dense(num_classes, activation='softmax',
                    kernel_regularizer=l2(1e-4))(fc)

    model = Model(inputs, outputs)
    return model
# ====================== 训练和评估模块 ======================
def train_model(X_train, y_train, X_val, y_val):
    """模型训练函数（包含 Downsampling、Focal Loss、调整超参）"""
    # 1. 下采样多数类（正常）以实现平衡
    idx_normal = np.where(y_train == 0)[0]
    idx_af     = np.where(y_train == 1)[0]

    # 将正常类随机下采样至与AF类一样多
    idx_normal_down = np.random.choice(idx_normal, size=len(idx_af), replace=False)
    new_idx = np.concatenate([idx_normal_down, idx_af])
    np.random.shuffle(new_idx)

    X_train_balanced = X_train[new_idx]
    y_train_balanced = y_train[new_idx]
    
    

    model = build_bilstm_attention_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # 检查类别平衡
    print(f"训练集原始类别分布: 正常={len(idx_normal)}, AF={len(idx_af)}")
    print(f"训练集下采样后类别分布: 正常={len(idx_af)}, AF={len(idx_af)}")

    f1_callback = F1ScoreCallback(validation_data=(X_val, y_val))

    # 创建回调
    early_stop = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_af_model_improved.keras', save_best_only=True, monitor='val_loss')
    reduce_lr  = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6, monitor='val_loss')
    f1_cb      = F1ScoreCallback(validation_data=(X_val, y_val))
    callbacks  = [early_stop, checkpoint, reduce_lr, f1_cb]

    history = model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print(f1_callback.f1_scores)
    return model, history, f1_callback

def evaluate_model(model, X_test, y_test):
    """模型评估函数"""
    # 预测
    y_pred = model.predict(X_test).argmax(axis=1)
    
    # 分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'AF']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'AF'])
    plt.yticks(tick_marks, ['Normal', 'AF'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    
    return y_pred

# ====================== 主执行流程 ======================
def main():
    # 配置参数 - 使用您的实际路径
    DATA_DIR = r'C:\Users\kara\Desktop\cpsc2021\Training_set_I'
    WINDOW_SIZE = 30  # 使用30个连续RR间期作为输入
    
    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据库路径不存在: {DATA_DIR}")
        print("请确认路径是否正确，并确保包含以下文件:")
        print(" - 04015.dat, 04015.hea, 04015.atr")
        print(" - 04043.dat, 04043.hea, 04043.atr")
        print(" - 等等...")
        return
    
    # 1. 加载数据
    records = load_mit_bih_af_records(DATA_DIR)
    print(f"在数据库中找到了 {len(records)} 条记录")
    print("前5条记录:", records[:5])
    
    # 2. 提取特征和标签
    all_X, all_y = [], []
    for i, record in enumerate(records):
        print(f"处理记录 {i+1}/{len(records)}: {record}")
        rr_intervals, rr_labels = extract_features(record, DATA_DIR)
        
        if rr_intervals is not None and len(rr_intervals) > 0:
            X, y = create_sequences(rr_intervals, rr_labels, WINDOW_SIZE)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                print(f"  提取了 {len(X)} 个序列, AF比例={np.mean(y):.4f}")
            else:
                print(f"  警告: 记录 {record} 没有足够的RR间期")
        else:
            print(f"  警告: 记录 {record} 没有提取到有效数据")
    
    # 合并所有记录
    if not all_X:
        print("错误: 没有提取到任何有效数据")
        return
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # 添加通道维度
    X = X[..., np.newaxis]
    
    print(f"\n数据集形状: X={X.shape}, y={y.shape}")
    print(f"AF比例: {np.mean(y):.4f} (总样本数: {len(y)})")
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"训练集AF比例: {np.mean(y_train):.4f}, 测试集AF比例: {np.mean(y_test):.4f}")
    
    # 4. 训练模型
    print("\n训练模型...")
    model, history, f1_callback = train_model(X_train, y_train, X_test, y_test)

    
    # 5. 评估模型
    print("\n评估模型...")
    evaluate_model(model, X_test, y_test)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # 6. 保存模型
    model.save('af_detection_model.keras')
    print("模型已保存为 af_detection_model.keras")

if __name__ == "__main__":
    main()
    