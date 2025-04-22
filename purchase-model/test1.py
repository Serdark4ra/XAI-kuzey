import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Purchase100Trainer:
    def __init__(self,
                 data_path,
                 model_type='fc',         # 'fc' or 'conv'
                 depth=3,
                 regularization='dropout',# 'l1', 'l2', 'dropout' or None
                 reg_constant=0.2,        # lowered dropout rate
                 test_size=0.1,
                 random_state=42):
        self.data_path      = data_path
        self.model_type     = model_type
        self.depth          = depth
        self.regularization = regularization
        self.reg_constant   = reg_constant
        self.test_size      = test_size
        self.random_state   = random_state

    def load_data(self):
        _, ext = os.path.splitext(self.data_path)
        ext = ext.lower()

        if ext == '.npz':
            with np.load(self.data_path, allow_pickle=True) as data:
                print("Loaded .npz with keys:", data.files)
                fkey = 'features' if 'features' in data.files else data.files[0]
                lkey = 'labels'   if 'labels'   in data.files else data.files[1]
                features   = data[fkey]
                labels_raw = data[lkey]

            if labels_raw.ndim > 1:
                self.y = labels_raw.astype(np.float32)
                num_classes = self.y.shape[1]
            else:
                labels_int = labels_raw.astype(int)
                num_classes = int(labels_int.max()) + 1
                self.y = to_categorical(labels_int, num_classes)
        else:
            raw        = np.loadtxt(self.data_path, delimiter=',', dtype=int)
            labels_int = raw[:, 0] - 1
            features   = raw[:, 1:]
            num_classes = int(labels_int.max()) + 1
            self.y     = to_categorical(labels_int, num_classes)

        self.X = features.astype(np.float32)
        self.normalize_features()
        self.input_shape = (self.X.shape[1],)
        print("input shape:", self.input_shape)
        self.num_classes = num_classes

    def normalize_features(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def split_data(self):
        strat = self.y.argmax(axis=1)
        '''
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            stratify=strat,
            random_state=self.random_state
        )
        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            stratify=strat,
            random_state=self.random_state
        )


    def make_fc(self):
        if   self.regularization == 'l1': k_reg = tf.keras.regularizers.L1L2(l1=self.reg_constant)
        elif self.regularization == 'l2': k_reg = tf.keras.regularizers.L1L2(l2=self.reg_constant)
        else:                             k_reg = None

        m = tf.keras.models.Sequential()
        m.add(tf.keras.Input(self.input_shape))

        if self.regularization == 'dropout':
            m.add(tf.keras.layers.Dropout(self.reg_constant))

        '''
        m.add(tf.keras.layers.Dense(256,
                                    activation='relu',  # improved from 'tanh'
                                    kernel_regularizer=k_reg))
        if self.regularization == 'dropout':
            m.add(tf.keras.layers.Dropout(self.reg_constant))
        '''



        for _ in range(self.depth):
            m.add(tf.keras.layers.Dense(128,
                                        activation='relu',  # improved from 'tanh'
                                        kernel_regularizer=k_reg))
            if self.regularization == 'dropout':
                m.add(tf.keras.layers.Dropout(self.reg_constant))

        m.add(tf.keras.layers.Dense(self.num_classes,
                                    activation='softmax',  # added softmax
                                    kernel_regularizer=k_reg))
        return m

    def make_conv(self):
        D = self.X.shape[1]
        H = W = int(np.sqrt(D))
        if H * W != D:
            raise ValueError(f"Can't reshape {D} dims into square image")
        df = 'channels_first'

        m = tf.keras.models.Sequential()
        m.add(tf.keras.Input((1, H, W)))

        for _ in range(self.depth):
            m.add(tf.keras.layers.Conv2D(32, 3, padding='same',
                                         data_format=df, activation='relu'))
            m.add(tf.keras.layers.Conv2D(32, 3,
                                         data_format=df, activation='relu'))
        m.add(tf.keras.layers.MaxPooling2D(2, data_format=df))

        for _ in range(self.depth):
            m.add(tf.keras.layers.Conv2D(64, 3, padding='same',
                                         data_format=df, activation='relu'))
            m.add(tf.keras.layers.Conv2D(64, 3,
                                         data_format=df, activation='relu'))
        m.add(tf.keras.layers.MaxPooling2D(2, data_format=df))

        if   self.regularization == 'l1': k_reg = tf.keras.regularizers.L1L2(l1=self.reg_constant)
        elif self.regularization == 'l2': k_reg = tf.keras.regularizers.L1L2(l2=self.reg_constant)
        else:                            k_reg = None

        if self.regularization == 'dropout':
            m.add(tf.keras.layers.Dropout(self.reg_constant))
        else:
            m.add(tf.keras.layers.Dropout(0.0))

        m.add(tf.keras.layers.Flatten())
        m.add(tf.keras.layers.Dense(512,
                                    activation='relu',
                                    kernel_regularizer=k_reg))
        if self.regularization == 'dropout':
            m.add(tf.keras.layers.Dropout(self.reg_constant))
        m.add(tf.keras.layers.Dense(self.num_classes,
                                    activation='softmax',  # added softmax
                                    kernel_regularizer=k_reg))
        return m

    def build_model(self):
        if self.model_type == 'fc':
            self.model = self.make_fc()
        else:
            D = self.X.shape[1]
            H = W = int(np.sqrt(D))
            self.X_train = self.X_train.reshape(-1, 1, H, W)
            #self.X_val   = self.X_val.reshape(-1, 1, H, W)
            self.X_test = self.X_test.reshape(-1, 1, H, W)
            self.model   = self.make_conv()

    def compile_and_train(self, epochs=60, batch_size=32):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',  # updated
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        self.history = self.model.fit(
            self.X_train, self.y_train,
            #validation_data=(self.X_val, self.y_val),
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks
        )

    def run(self, epochs=60, batch_size=32):  # use 60 by default
        self.load_data()
        self.split_data()
        self.build_model()
        print(self.model.summary())
        self.compile_and_train(epochs, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    required=True,
                        help=".npz or CSV file path")
    parser.add_argument("--model_type",   choices=['fc','conv'], default='fc')
    parser.add_argument("--depth",        type=int, default=3)
    parser.add_argument("--regularization", choices=['l1','l2','dropout',None],
                        default='dropout')
    parser.add_argument("--reg_constant", type=float, default=0.2)
    parser.add_argument("--epochs",       type=int, default=60)
    parser.add_argument("--batch_size",   type=int, default=32)
    args = parser.parse_args()

    trainer = Purchase100Trainer(
        data_path      = args.data_path,
        model_type     = args.model_type,
        depth          = args.depth,
        regularization = args.regularization,
        reg_constant   = args.reg_constant
    )
    trainer.run(epochs=args.epochs, batch_size=args.batch_size)

    trainer.model.evaluate(trainer.X_test, trainer.y_test)

    trainer.model.save_weights('purchase100_model.weights.h5')
