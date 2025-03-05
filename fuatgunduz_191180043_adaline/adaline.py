class AdalineModel:
    def __init__(self, learning_rate=0.5):
        """
        Adaline modelini varsayılan öğrenme oranı (learning rate) ile başlatır.
        """
        self.w1 = 0.3  # Başlangıç ağırlığı (x1 için)
        self.w2 = 0.2  # Başlangıç ağırlığı (x2 için)
        self.threshold = 0.1  # Başlangıç eşik değeri (threshold)
        self.alpha = learning_rate  # Öğrenme oranı
    
    def predict(self, x1, x2):
        """
        Net girdiyi (NET) hesaplar ve sonucu döner.
        """
        net = self.w1 * x1 + self.w2 * x2 + self.threshold
        return 1 if net > 0 else -1
    
    def train(self, x1, x2, expected_output):
        """
        Adaline modelini öğrenme kuralını kullanarak eğitir.
        """
        net = self.w1 * x1 + self.w2 * x2 + self.threshold
        current_output = 1 if net > 0 else -1
        error = expected_output - current_output
        
        # Ağırlıklar ve eşik değerini güncelle
        self.w1 += self.alpha * error * x1
        self.w2 += self.alpha * error * x2
        self.threshold += self.alpha * error
