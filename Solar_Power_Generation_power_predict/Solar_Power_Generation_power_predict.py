from tensorflow.data  import Dataset
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
import tensorflow as tf
import time
import random
import math

#超參數設定
data_contain_days    = 3
data_predict_days    = 2
epochs               = 10
batch_size           = 32
learning_rate        = 0.001
weather_data_path    = "Plant_1_Weather_Sensor_Data.csv"
generation_data_path = "Plant_1_Generation_Data.csv"

class power_generation_predict():
    def __init__(self, weather_data_path, generation_data_path, data_contain_days, data_predict_days, epochs, batch_size, learning_rate):
        self.weather_data_path                  = weather_data_path 
        self.generation_data_path               = generation_data_path
        self.data_contain_days                  = data_contain_days
        self.data_predict_days                  = data_predict_days
        self.epochs                             = epochs
        self.batch_size                         = batch_size
        self.learning_rate                      = learning_rate
        self.model                              = None
        self.weather_data                       = None
        self.generation_data                    = None
        self.generation_data_split_by_inverter  = None
        self.weather_data_match_by_inveter_date = None
        self.train_dataset                      = None
        self.train_data_steps_each_epoch        = None
        self.test_dataset                       = None
        self.test_data_steps_each_epoch         = None
        self.data_max                           = None
        self.data_min                           = None

    def load_csv_file(self):
        #匯入資料
        self.weather_data   = tf.io.read_file(self.weather_data_path)
        #以分隔符號將資料分割並移除header
        self.weather_data   = tf.strings.split(self.weather_data, "\n")
        self.weather_data   = tf.convert_to_tensor(self.weather_data.numpy()[1:])
        csv_column_datatype = [str(), int(), str(), float(), float(), float()]
        self.weather_data   = tf.io.decode_csv(self.weather_data, csv_column_datatype)
        weatherdata_max     = []
        weatherdata_min     = []
        for index in range(3, len(self.weather_data)):
            weatherdata_max.append(tf.math.reduce_max(self.weather_data[index]))
            weatherdata_min.append(tf.math.reduce_min(self.weather_data[index]))

        #匯入資料
        self.generation_data = tf.io.read_file(self.generation_data_path)
        #以分隔符號將資料分割並移除header
        self.generation_data = tf.strings.split(self.generation_data, "\r\n")
        self.generation_data = tf.convert_to_tensor(self.generation_data.numpy()[1:])
        csv_column_datatype  = [str(), int(), str(), float(), float(), float(), float()]
        self.generation_data = tf.io.decode_csv(self.generation_data, csv_column_datatype)
        generation_data_max  = []
        generation_data_min  = []
        for index in range(3, 5):
            generation_data_max.append(tf.math.reduce_max(self.generation_data[index]))
            generation_data_min.append(tf.math.reduce_min(self.generation_data[index]))

        self.data_max = tf.convert_to_tensor(generation_data_max + weatherdata_max)
        self.data_min = tf.convert_to_tensor(generation_data_min + weatherdata_min)

    def change_generation_datetime_format(self):
        #修改時間格式讓兩個資料集能相符
        generation_datetime = self.generation_data[0]
        split_tensor        = tf.strings.split(generation_datetime, " ")
        g_date, g_time      = tf.split(value = split_tensor.numpy(), num_or_size_splits = 2, axis = 1)
        split_date          = tf.strings.split(tf.squeeze(g_date), "-")
        day, month, year    = tf.split(value = split_date.numpy(), num_or_size_splits = 3, axis = 1)
        split_date          = tf.strings.join([year, tf.strings.join([month, day], "-")], "-")
        split_time          = tf.strings.join([g_time, ":00"])
        generation_datetime = tf.squeeze(tf.strings.join([split_date, split_time]," "))
        self.generation_data[0]  = generation_datetime

    def split_generation_data_by_inverter_and_match_waether_data_by_inverter_datetime(self):
        #將發電資料根據inverter分割
        inverter_id         = tf.unique(self.generation_data[2])
        inverter_id_label   = inverter_id.y
        inverter_id         = inverter_id.idx

        self.generation_data_split_by_inverter  = []
        self.weather_data_match_by_inveter_date = []
        #分割發電資料並和天氣資料配對
        for index in range(inverter_id_label.shape[0]):
            inverter_index        = tf.equal(inverter_id, index)
            split_generation_date = tf.boolean_mask(self.generation_data[0], inverter_index)   
            weather_date_match_index_list = []
            inveter_data_keep_index_list  = []
            for match_index in range(split_generation_date.shape[0]):
                self.search_match_date(split_generation_date, self.weather_data[0], weather_date_match_index_list,
                                       inveter_data_keep_index_list, match_index)

            weather_date_match_index_list = tf.convert_to_tensor(weather_date_match_index_list)
            inveter_data_keep_index_list  = tf.convert_to_tensor(inveter_data_keep_index_list)
    
            split_generation_data = []
            #只保留要分析的部分
            for split_index in range(3, 5):
                split_tensor = tf.boolean_mask(self.generation_data[split_index], inverter_index)
                split_tensor = tf.convert_to_tensor(split_tensor.numpy()[inveter_data_keep_index_list.numpy()])
                split_generation_data.append(split_tensor)
            self.generation_data_split_by_inverter.append(tf.transpose(tf.convert_to_tensor(split_generation_data)))

            split_weather_data = []
            #只保留要分析的部分
            for split_index in range(3,len(self.weather_data)):
                split_tensor = tf.convert_to_tensor(self.weather_data[split_index].numpy()[weather_date_match_index_list.numpy()])
                split_weather_data.append(split_tensor)
            self.weather_data_match_by_inveter_date.append(tf.transpose(tf.convert_to_tensor(split_weather_data)))

    def search_match_date(self, split_generation_date, weather_date, weather_date_match_index_list,
                          inveter_data_keep_index_list, index):
        match_index = tf.math.equal(split_generation_date, weather_date[index])
        match_index = tf.where(match_index)
        if match_index.shape[0] == 1:
            weather_date_match_index_list.append(tf.squeeze(match_index))
            inveter_data_keep_index_list.append(index)

    def creat_train_data_and_test_data(self):
        #資料是以15分鐘為取樣頻率
        #一天大約有96筆
        number_of_data_each_day = 96
        #每筆資料間隔一小時
        data_shift_range = 4
        train_data = []
        test_data  = []
        for inverter_index in range(len(self.generation_data_split_by_inverter)):
            data_list   = []
            start_index = 0
            max_index   = self.generation_data_split_by_inverter[inverter_index].shape[0]
            generation_data = tf.concat(values = [self.generation_data_split_by_inverter[inverter_index], self.weather_data_match_by_inveter_date[inverter_index]],
                                        axis = 1)
            keep_loop = True
            while keep_loop:
                if start_index + (number_of_data_each_day * (self.data_contain_days + self.data_predict_days)) <= max_index:
                    end_index = start_index + (number_of_data_each_day * (self.data_contain_days + self.data_predict_days))
                else:
                    start_index = max_index - (number_of_data_each_day * (self.data_contain_days + self.data_predict_days))
                    end_index   = max_index
                    keep_loop = False

                data_list.append(generation_data[start_index:end_index, :])
                start_index += data_shift_range 
                
            data_list  = random.sample(data_list, len(data_list))
            train_data += data_list[0:math.ceil(len(data_list) * 0.8)]
            test_data  += data_list[math.ceil(len(data_list) * 0.8):len(data_list)]
        return tf.convert_to_tensor(train_data), tf.convert_to_tensor(test_data)

    def recurrent_data_process(self, recurrent_data):
        recurrent_data= (recurrent_data - self.data_min) / (self.data_max - self.data_min) 
        number_of_data_each_day = 96

        recurrent_data, predict_data = tf.split(recurrent_data,
                                                [number_of_data_each_day * self.data_contain_days,
                                                 number_of_data_each_day * self.data_predict_days],
                                                 0)
        predict_dc_power, predict_ac_power, predict_weather_data = tf.split(predict_data, [1, 1, predict_data.shape[1] - 2], 1)
        #predict_dc_power  = tf.squeeze(predict_dc_power)
        #predict_ac_power  = tf.squeeze(predict_ac_power)
        #a_day_dc_power    = tf.math.reduce_sum(predict_dc_power[0:number_of_data_each_day])
        #two_days_dc_power = tf.math.reduce_sum(predict_dc_power)
        #a_day_ac_power    = tf.math.reduce_sum(predict_ac_power[0:number_of_data_each_day])
        #two_days_ac_power = tf.math.reduce_sum(predict_ac_power)
        a_day_dc_power    = predict_dc_power[0]
        two_days_dc_power = predict_dc_power[1]
        a_day_ac_power    = predict_ac_power[0]
        two_days_ac_power = predict_ac_power[1]

        return (recurrent_data, (a_day_dc_power, a_day_dc_power, two_days_dc_power, two_days_ac_power))

    def recurrent_label_1_process(self, recurrent_data):
        recurrent_data= (recurrent_data - self.data_min) / (self.data_max - self.data_min)
        number_of_data_each_day = 96
        recurrent_data, predict_data = tf.split(recurrent_data,
                                                [number_of_data_each_day * self.data_contain_days,
                                                 number_of_data_each_day * self.data_predict_days],
                                                 0)
        predict_dc_power, predict_ac_power, predict_weather_data = tf.split(predict_data, [1, 1, predict_data.shape[1] - 2], 1)
        predict_dc_power  = tf.squeeze(predict_dc_power)
        #predict_ac_power  = tf.squeeze(predict_ac_power)
        #a_day_dc_power    = tf.math.reduce_sum(predict_dc_power[0:number_of_data_each_day])
        #two_days_dc_power = tf.math.reduce_sum(predict_dc_power)
        #a_day_ac_power    = tf.math.reduce_sum(predict_ac_power[0:number_of_data_each_day])
        #two_days_ac_power = tf.math.reduce_sum(predict_ac_power)
        a_day_dc_power    = predict_dc_power[0]
        #two_days_dc_power = predict_dc_power[1]
        #a_day_ac_power    = predict_ac_power[0]
        #two_days_ac_power = predict_ac_power[1]

        return a_day_dc_power

    def recurrent_label_2_process(self, recurrent_data):
        recurrent_data= (recurrent_data - self.data_min) / (self.data_max - self.data_min)
        number_of_data_each_day = 96
        recurrent_data, predict_data = tf.split(recurrent_data,
                                                [number_of_data_each_day * self.data_contain_days,
                                                 number_of_data_each_day * self.data_predict_days],
                                                 0)
        predict_dc_power, predict_ac_power, predict_weather_data = tf.split(predict_data, [1, 1, predict_data.shape[1] - 2], 1)
        #predict_dc_power  = tf.squeeze(predict_dc_power)
        predict_ac_power  = tf.squeeze(predict_ac_power)
        #a_day_dc_power    = tf.math.reduce_sum(predict_dc_power[0:number_of_data_each_day])
        #two_days_dc_power = tf.math.reduce_sum(predict_dc_power)
        #a_day_ac_power    = tf.math.reduce_sum(predict_ac_power[0:number_of_data_each_day])
        #two_days_ac_power = tf.math.reduce_sum(predict_ac_power)
        #a_day_dc_power    = predict_dc_power[0]
        #two_days_dc_power = predict_dc_power[1]
        a_day_ac_power    = predict_ac_power[0]
        #two_days_ac_power = predict_ac_power[1]

        return a_day_ac_power

    def recurrent_label_3_process(self, recurrent_data):
        recurrent_data= (recurrent_data - self.data_min) / (self.data_max - self.data_min)
        number_of_data_each_day = 96
        recurrent_data, predict_data = tf.split(recurrent_data,
                                                [number_of_data_each_day * self.data_contain_days,
                                                 number_of_data_each_day * self.data_predict_days],
                                                 0)
        predict_dc_power, predict_ac_power, predict_weather_data = tf.split(predict_data, [1, 1, predict_data.shape[1] - 2], 1)
        predict_dc_power  = tf.squeeze(predict_dc_power)
        #predict_ac_power  = tf.squeeze(predict_ac_power)
        #a_day_dc_power    = tf.math.reduce_sum(predict_dc_power[0:number_of_data_each_day])
        #two_days_dc_power = tf.math.reduce_sum(predict_dc_power)
        #a_day_ac_power    = tf.math.reduce_sum(predict_ac_power[0:number_of_data_each_day])
        #two_days_ac_power = tf.math.reduce_sum(predict_ac_power)
        #a_day_dc_power    = predict_dc_power[0]
        two_days_dc_power = predict_dc_power[1]
        #a_day_ac_power    = predict_ac_power[0]
        #two_days_ac_power = predict_ac_power[1]

        return two_days_dc_power

    def recurrent_label_4_process(self, recurrent_data):
        recurrent_data= (recurrent_data - self.data_min) / (self.data_max - self.data_min)
        number_of_data_each_day = 96
        recurrent_data, predict_data = tf.split(recurrent_data,
                                                [number_of_data_each_day * self.data_contain_days,
                                                 number_of_data_each_day * self.data_predict_days],
                                                 0)
        predict_dc_power, predict_ac_power, predict_weather_data = tf.split(predict_data, [1, 1, predict_data.shape[1] - 2], 1)
        #predict_dc_power  = tf.squeeze(predict_dc_power)
        predict_ac_power  = tf.squeeze(predict_ac_power)
        #a_day_dc_power    = tf.math.reduce_sum(predict_dc_power[0:number_of_data_each_day])
        #two_days_dc_power = tf.math.reduce_sum(predict_dc_power)
        #a_day_ac_power    = tf.math.reduce_sum(predict_ac_power[0:number_of_data_each_day])
        #two_days_ac_power = tf.math.reduce_sum(predict_ac_power)
        #a_day_dc_power    = predict_dc_power[0]
        #two_days_dc_power = predict_dc_power[1]
        #a_day_ac_power    = predict_ac_power[0]
        two_days_ac_power = predict_ac_power[1]

        return two_days_ac_power

    def creat_recurrent_dataset(self, data_list, shuffle = True):
        data_size = data_list.shape[0]
        dataset   = Dataset.from_tensor_slices(data_list)
        
        dataset = dataset.map(self.recurrent_data_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(data_list.shape[0]).repeat()
        else:
            dataset = dataset.repeat()
        
        dataset         = dataset.batch(self.batch_size)
        step_each_epoch = math.ceil(data_size / self.batch_size)

        return dataset, step_each_epoch

    def creat_dataset(self):
        self.load_csv_file()
        self.change_generation_datetime_format()
        self.split_generation_data_by_inverter_and_match_waether_data_by_inverter_datetime()

        train_data, test_data = self.creat_train_data_and_test_data()
        
        self.train_dataset, self.train_data_steps_each_epoch = self.creat_recurrent_dataset(train_data, shuffle = True)
        self.test_dataset, self.test_data_steps_each_epoch   = self.creat_recurrent_dataset(test_data, shuffle = True)

    def creat_model(self):
        input_layer  = Input(shape = (None, 5))
        #gru_layer_1  = layers.GRU(8, activation = "relu", dropout = 0.1, recurrent_dropout = 0.4, return_sequences = True)(input_layer)
        #gru_layer_2  = layers.GRU(12, activation = "relu", dropout = 0.1, recurrent_dropout = 0.4, return_sequences = True)(gru_layer_1)
        #gru_layer_3  = layers.GRU(16, activation = "relu", dropout = 0.1, recurrent_dropout = 0.4, return_sequences = True)(gru_layer_2)
        #gru_layer_4  = layers.GRU(16, activation = "relu", dropout = 0.1, recurrent_dropout = 0.4, return_sequences = True)(gru_layer_3)
        #gru_layer_5  = layers.GRU(10, activation = "relu", dropout = 0.1, recurrent_dropout = 0.4)(gru_layer_4)
        gru_layer_1  = layers.GRU(8, activation = "relu", return_sequences = True)(input_layer)
        gru_layer_2  = layers.GRU(12, activation = "relu", return_sequences = True)(gru_layer_1)
        gru_layer_3  = layers.GRU(16, activation = "relu", return_sequences = True)(gru_layer_2)
        gru_layer_4  = layers.GRU(16, activation = "relu", return_sequences = True)(gru_layer_3)
        gru_layer_5  = layers.GRU(10, activation = "relu")(gru_layer_4)
        output_1     = layers.Dense(1, name = "dc_power_in_15_minute")(gru_layer_5)
        output_2     = layers.Dense(1, name = "ac_power_in_15_minute")(gru_layer_5)
        output_3     = layers.Dense(1, name = "dc_power_in_30_minute")(gru_layer_5)
        output_4     = layers.Dense(1, name = "ac_power_in_30_minute")(gru_layer_5)
        
        self.model = Model(input_layer, [output_1, output_2, output_3, output_4])

    def model_training(self):
        self.creat_dataset()
        self.creat_model()

        self.model.summary()

        self.model.compile(optimizer =  tf.keras.optimizers.Adam(self.learning_rate), loss = "mse")
        #建立callback
        callback_path              = "model_callback_output/model_weights"
        tensorboard_path           = "tensorboard_output" 
        model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath = callback_path, monitor = "val_loss", save_best_only = True,
                                                                        save_weights_only = True)
        tensorboard_callback       = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path, histogram_freq = 1)
        #history = self.model.fit(x = self.train_dataset, steps_per_epoch = self.train_data_steps_each_epoch, epochs = self.epochs)
        history = self.model.fit(x = self.train_dataset, steps_per_epoch = self.train_data_steps_each_epoch, epochs = self.epochs,
                                  validation_data = self.test_dataset, validation_steps = self.test_data_steps_each_epoch, 
                                  callbacks = [model_check_point_callback, tensorboard_callback])

generation_power_predict_model = power_generation_predict(weather_data_path, generation_data_path,
                                                          data_contain_days, data_predict_days, epochs, batch_size, learning_rate)
generation_power_predict_model.model_training()

