import numpy as np
import os
import matplotlib.pyplot as plt
import csv


class DataModel:

    def __init__(self, data):
        self.data = np.asarray(data)
        self.fft = np.asarray(self.__fft_mag())
        self.a_weighted = np.asarray(self.__a_weight())
        self.snr_normalized = self.__snr(aweight=False)
        self.snr_aweight = self.__snr(aweight=True)

    def __fft_mag(self):
        """
        Returns the normalized magnitude FFT from the data given in.
        :param Fs:
        :return:
        """
        data_fft = self.data
        data_fft = np.fft.fft(data_fft)
        # data_fft = np.abs(data_fft)
        data_fft = data_fft[range(len(data_fft) // 2)]              #see only one peak
        data_fft = np.abs(data_fft / (len(data_fft // 2)))          #normalize signal
        return data_fft

    def __a_weight(self, fs=192000):
        xlen = len(self.data)
        num_frequency_bins = np.ceil((xlen + 1) / 2)
        f = np.arange(0, (num_frequency_bins - 1)) * fs / xlen

        c1 = 3.5041384 * (10 ** 16)
        c2 = 20.598997 ** 2
        c3 = 107.65265 ** 2
        c4 = 737.86223 ** 2
        c5 = 12194.217 ** 2

        f = np.power(f, 2)

        num = c1*np.power(f, 4)
        den = (np.power(c2 + f, 2))
        den = np.multiply(den, (c3 + f))
        den = np.multiply(den, (c4 + f))
        den = np.multiply(den, np.power(c5 + f, 2))

        a_weighted = np.divide(num, den)

        a_weighted_array = np.abs(np.multiply(a_weighted, self.get_fft()))
        a_weighted_array = np.delete(a_weighted_array, 0)
        return a_weighted_array

    def __snr(self, aweight=True):
        def find_fundamental(data):
            max_idx = np.argmax(data[1:])
            max_value = data[max_idx]
            return max_value, max_idx
        # checking signal
        if aweight:
            data = self.get_a_weight()
        else:
            data = self.get_fft()
        fund_indx = find_fundamental(data)[1]
        sum_of_squares = 0
        rng = data[1:]
        for i in range(len(rng)):
            # if i not in (fund_indx-1, fund_indx, fund_indx+1):
            if i != fund_indx:
                sum_of_squares += np.power(data[i], 2)
        sum_of_squares = np.sqrt(sum_of_squares)
        snr = sum_of_squares / find_fundamental(data)[1]
        snr = 20 * np.log10(snr)
        fundamental = 20 * np.log10(find_fundamental(data)[1])
        return fundamental - snr

    def get_fft(self):
        return self.fft

    def get_data(self):
        return np.asarray(self.data)

    def get_a_weight(self):
        return self.a_weighted

    def get_a_weight_log(self):
        return self.get_log(self.get_a_weight())

    def get_log_fft(self):
        return self.get_log(self.get_fft())

    def get_snr(self, a_weighted):
        if a_weighted:
            return self.snr_aweight
        else:
            return self.snr_normalized

    @staticmethod
    def get_log(data):
        return 20.*np.log10(np.abs(data))


class View:
    def __init__(self, model):
        self.model = model

    def plot_data(self):
        print("SNR (dB, A-Weighted): {}".format(self.model.get_snr(True)))
        print("SNR (dB, Normalized): {}".format(self.model.get_snr(False)))
        fig1 = plt.figure(1)
        ax1_1 = fig1.add_subplot(211)
        ax2_1 = fig1.add_subplot(212)
        self.plot_fft(ax1_1)
        self.plot_waveform(ax2_1)
        fig2 = plt.figure(2)
        ax1_2 = fig2.add_subplot(211)
        self.plot_a_weight(ax1_2)

    def plot_fft(self, ax):
        fft_data = self.model.get_log_fft()
        print(len(fft_data))
        ax.plot(np.arange(len(fft_data)), fft_data)
        ax.set_xlim(0, 20000)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("FFT (normalized)")
        ax.grid(True)

    def plot_waveform(self, ax, fs=192000):
        data = np.asarray(self.model.get_data())
        # max_value = max(data)
        # min_value = min(data)
        time = 1/fs*len(data)
        time_divisions = time / fs
        num_periods = int(input('Enter how many periods you want to plot in the time domain: '))
        def calculate_period_length(data):
            ref = data[0]

        # time_divisions *= (num_periods * )
        range_values = np.arange(0, time, 1/fs)
        ax.plot(range_values, data)
        # ax.set_ylim(min_value * 1.5, max_value * 1.5)
        ax.set_xlim(0, .025*time)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Waveform")
        ax.grid(True)
        plt.show()

    def plot_a_weight(self, ax):
        a_weight_data = self.model.get_a_weight_log()
        ax.plot(np.arange(len(a_weight_data)), a_weight_data)
        ax.set_xlim(0, 20000)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("FFT (normalized)")
        ax.grid(True)
        plt.show()


class Controller:
    def __init__(self):
        self.filePath = None
        self.data = []
        sel = 0
        # input validation
        while not sel:
            try:
                sel = int(input('Select 1 for noisy sine wave input, 2 for datafile input: '))
                if sel not in (1, 2):
                    raise ValueError
            except ValueError:
                print("Unable to parse input. Try again.")
                sel = int(input('Select 1 for noisy sine wave input, 2 for datafile input: '))
        if sel == 1:
            self.sine_input()
        else:
            self.data_input()
        self.model = DataModel(self.data)
        self.view = View(self.model)

    def sine_input(self, fs=48000):
        amplitude = 1 * np.sqrt(2)
        f = 2000
        samples = fs
        x = np.arange(samples)
        self.data = np.sin(2 * np.pi * f * x / fs)
        self.data = np.asarray(amplitude * self.data)
        noise = np.random.normal(0, (.05 * np.sqrt(2)), self.data.shape)
        self.data = np.asarray([noise[i] + self.data[i] for i in range(len(noise))])

    def data_input(self):
        self.filePath, self.data = self.find_input()
        if (len(self.data) > 0) and self.filePath:
            print("Data entered successfully!")

    @staticmethod
    def find_input():
        subdir = "Data"
        path = os.path.normpath(os.path.join(os.getcwd(), subdir))
        files = os.listdir(path)
        print("Current files: ")
        print("-------------------------------------------------------------")
        for f in files:
            print(f)
        print("-------------------------------------------------------------")
        file = input("Enter the file you want to read from: ")
        data = None
        prev_path = path
        while data is None:
            try:
                path = os.path.normpath(os.path.join(path, file))
                extension = file.split(".")
                if len(extension) > 1:
                    if extension[1] == "csv":
                        # do csv stuff here
                        data = []
                        with open(path) as csvfile:
                            read_csv = csv.reader(csvfile, delimiter=',')
                            for row in read_csv:
                                if row:
                                    if '#' not in row[0]:
                                        data.append(row[0])
                            data = np.asarray(data)
                    elif extension[1] == "txt":
                        # do text stuff here
                        data = np.loadtxt(path)
                    else:
                        raise IOError("Unsupported file type selected.")
                else:
                    raise IOError("Improper file name entered.")
            except OSError:
                print('Unable to find file.')
                file = input("Enter the file name that you want to read from: ")
                path = prev_path
            except IOError:
                file = input("Enter the file name that you want to read from: ")
                path = prev_path
        return path, data

    def get_model_data(self):
        print("Getting model data...")
        print(self.model.get_fft())

    def run(self):
        self.view.plot_data()
        # self.get_model_data()


if __name__ == '__main__':
    c = Controller()
    c.run()
