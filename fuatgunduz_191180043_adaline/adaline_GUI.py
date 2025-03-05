import tkinter as tk
from adaline import AdalineModel

# Adaline modeli için arayüz sınıfı
class AdalineInterface:
    def __init__(self, root):
        # Adaline modelini başlat
        self.model = AdalineModel()
        self.root = root
        self.root.title("Adaline Model Interface")

        # Öğrenme oranı etiketi ve giriş kutusu
        tk.Label(root, text="Learning Rate:").grid(row=0, column=0)
        self.learning_rate_entry = tk.Entry(root)
        self.learning_rate_entry.insert(0, str(self.model.alpha))
        self.learning_rate_entry.grid(row=0, column=1)

        # w1 ağırlığı etiketi ve giriş kutusu
        tk.Label(root, text="w1:").grid(row=1, column=0)
        self.w1_entry = tk.Entry(root)
        self.w1_entry.insert(0, str(self.model.w1))
        self.w1_entry.grid(row=1, column=1)

        # w2 ağırlığı etiketi ve giriş kutusu
        tk.Label(root, text="w2:").grid(row=2, column=0)
        self.w2_entry = tk.Entry(root)
        self.w2_entry.insert(0, str(self.model.w2))
        self.w2_entry.grid(row=2, column=1)

        # Eşik değeri etiketi ve giriş kutusu
        tk.Label(root, text="Threshold:").grid(row=3, column=0)
        self.threshold_entry = tk.Entry(root)
        self.threshold_entry.insert(0, str(self.model.threshold))
        self.threshold_entry.grid(row=3, column=1)

        # x1 girdisi etiketi ve giriş kutusu
        tk.Label(root, text="x1:").grid(row=4, column=0)
        self.x1_entry = tk.Entry(root)
        self.x1_entry.insert(0, "1")
        self.x1_entry.grid(row=4, column=1)

        # x2 girdisi etiketi ve giriş kutusu
        tk.Label(root, text="x2:").grid(row=5, column=0)
        self.x2_entry = tk.Entry(root)
        self.x2_entry.insert(0, "0")
        self.x2_entry.grid(row=5, column=1)

        # Beklenen çıktı etiketi ve giriş kutusu
        tk.Label(root, text="Expected Output:").grid(row=6, column=0)
        self.expected_output_entry = tk.Entry(root)
        self.expected_output_entry.insert(0, "-1")
        self.expected_output_entry.grid(row=6, column=1)

        # Eğit butonu
        self.train_button = tk.Button(root, text="Train", command=self.train_model)
        self.train_button.grid(row=7, columnspan=2)

        # Sonuçları göstermek için etiket
        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=8, columnspan=2)

    # Modeli eğitmek için fonksiyon
    def train_model(self):
        try:
            # Kullanıcıdan alınan değerleri modele aktar
            self.model.alpha = float(self.learning_rate_entry.get())
            self.model.w1 = float(self.w1_entry.get())
            self.model.w2 = float(self.w2_entry.get())
            self.model.threshold = float(self.threshold_entry.get())
            x1 = float(self.x1_entry.get())
            x2 = float(self.x2_entry.get())
            expected_output = int(self.expected_output_entry.get())

            iteration = 1  # Iterasyon sayacı
            result = ""  # Sonuçları depolamak için değişken
            error = 1  # Hata başlangıç değeri

            # Hata sıfır olana kadar eğitimi sürdür
            while error != 0:
                # Güncellemeler öncesi başlangıç ağırlıkları ve eşik
                initial_w1, initial_w2, initial_threshold = self.model.w1, self.model.w2, self.model.threshold
                self.model.train(x1, x2, expected_output)

                # Net giriş ve mevcut çıktıyı hesapla
                net = initial_w1 * x1 + initial_w2 * x2 + initial_threshold
                current_output = 1 if net > 0 else -1
                error = expected_output - current_output

                # Sonuçları birleştir
                result += (f"{iteration}st Iteration:\n"
                           f"Initial Weights: w1={initial_w1}, w2={initial_w2}\n"
                           f"Initial Threshold: {initial_threshold}\n"
                           f"After training:\n"
                           f"Updated Weights: w1={self.model.w1}, w2={self.model.w2}\n"
                           f"Updated Threshold: {self.model.threshold}\n"
                           f"Error: {error}\n\n")
                iteration += 1

            # Sonuçları arayüzde göster
            self.result_label.config(text=result)
        except ValueError:
            # Hatalı giriş durumunda hata mesajı
            self.result_label.config(text="Invalid input. Please enter numeric values.")

# Arayüzü başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = AdalineInterface(root)
    root.mainloop()
