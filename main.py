import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import os
import re

class SentimentAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân tích thái độ Văn bản")
        self.root.geometry("800x700")
        
        # Initialize model components
        self.loaded_model = None
        self.phobert_encoder = None
        self.tokenizer = None
        self.label_encoder = None
        self.MAX_LEN = None
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text input area
        ttk.Label(main_frame, text="Nhập văn bản cần phân tích:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.text_input = scrolledtext.ScrolledText(main_frame, height=6, width=70)
        self.text_input.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        self.text_input.insert(tk.END, "Sản phẩm này dùng cũng ổn, không quá tệ.")
        
        # Analyze button
        self.analyze_btn = ttk.Button(main_frame, text="Phân tích thái độ", command=self.analyze_sentiment)
        self.analyze_btn.grid(row=2, column=0, pady=(0, 10))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Kết quả", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.result_label = ttk.Label(results_frame, text="Chưa có kết quả")
        self.result_label.grid(row=0, column=0, sticky=tk.W)
        
        self.confidence_label = ttk.Label(results_frame, text="")
        self.confidence_label.grid(row=1, column=0, sticky=tk.W)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(main_frame, text="Biểu đồ xác suất", padding="10")
        chart_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
    
    def load_models(self):
        try:
            print("--- Đang tải model và thông tin cần thiết ---")
            
            # Load keras model
            model_path = os.path.join('trained_model', 'sentiment_analyzer_best_model.keras')
            if os.path.exists(model_path):
                self.loaded_model = keras.models.load_model(model_path)
                print(f"Đã load model từ: {model_path}")
            else:
                print(f"LỖI: Không tìm thấy model tại {model_path}")
                messagebox.showerror("Lỗi", f"Không tìm thấy model tại {model_path}")
                return
            
            # Load model info
            model_info_path = os.path.join('trained_model', 'sentiment_analyzer_model_info.pkl')
            if os.path.exists(model_info_path):
                with open(model_info_path, 'rb') as f:
                    model_info = pickle.load(f)
                self.MAX_LEN = model_info.get('max_len_sequence', 128)
                print(f"MAX_LEN: {self.MAX_LEN}")
            else:
                self.MAX_LEN = 128
                print("Sử dụng MAX_LEN mặc định: 128")
            
            # Load label encoder
            label_encoder_path = os.path.join('trained_model', 'sentiment_analyzer_label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"Đã load label_encoder từ: {label_encoder_path}")
            else:
                print(f"LỖI: Không tìm thấy label_encoder tại {label_encoder_path}")
                messagebox.showerror("Lỗi", f"Không tìm thấy label encoder tại {label_encoder_path}")
                return
            
            # Load PhoBERT tokenizer and model
            phobert_tokenizer_name = "vinai/phobert-base-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(phobert_tokenizer_name)
            self.phobert_encoder = TFAutoModel.from_pretrained(phobert_tokenizer_name, from_pt=True)
            print("Đã tải xong PhoBERT tokenizer và encoder model.")
            
            messagebox.showinfo("Thành công", "Đã tải xong tất cả model!")
            
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            messagebox.showerror("Lỗi", f"Lỗi khi tải model: {e}")
    
    def preprocess_text_for_prediction(self, text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def predict_sentiment(self, text_input):
        if not all([self.loaded_model, self.phobert_encoder, self.tokenizer, self.label_encoder]):
            return None
        
        processed_text = self.preprocess_text_for_prediction(text_input)
        encoded_inputs = self.tokenizer(
            [processed_text],
            padding='max_length',
            truncation=True,
            max_length=self.MAX_LEN,
            return_tensors='tf'
        )
        
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        
        physical_devices = tf.config.list_physical_devices('GPU')
        device_to_use = '/GPU:0' if physical_devices else '/CPU:0'
        
        with tf.device(device_to_use):
            phobert_outputs = self.phobert_encoder(input_ids=input_ids, attention_mask=attention_mask, training=False)
        
        input_embeddings_for_model = phobert_outputs.last_hidden_state
        prediction_probabilities = self.loaded_model.predict(input_embeddings_for_model, verbose=0)
        
        predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]
        predicted_label_text = self.label_encoder.inverse_transform([predicted_class_index])[0]
        confidence_score = float(prediction_probabilities[0][predicted_class_index])
        
        return {
            'text_input': text_input,
            'processed_text': processed_text,
            'label': predicted_label_text,
            'confidence': confidence_score,
            'prediction_vector': prediction_probabilities[0].tolist()
        }
    
    def analyze_sentiment(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập văn bản!")
            return
        
        if not all([self.loaded_model, self.phobert_encoder, self.tokenizer, self.label_encoder]):
            messagebox.showerror("Lỗi", "Model chưa được tải. Vui lòng khởi động lại ứng dụng.")
            return
        
        try:
            result = self.predict_sentiment(input_text)
            
            if result:
                self.result_label.config(text=f"Nhãn dự đoán: {result['label']}")
                self.confidence_label.config(text=f"Độ tin cậy: {result['confidence']:.2%}")
                
                # Update chart
                self.update_chart(result['prediction_vector'])
            else:
                messagebox.showerror("Lỗi", "Không thể phân tích văn bản.")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi phân tích: {e}")
    
    def update_chart(self, prediction_vector):
        self.ax.clear()
        
        labels = self.label_encoder.classes_
        colors = ['#FF9999', '#ADD8E6', '#90EE90']  # Colors for NEG, NEU, POS
        
        bars = self.ax.bar(labels, prediction_vector, color=colors)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2%}', ha='center', va='bottom')
        
        self.ax.set_title('Biểu đồ xác suất dự đoán cho từng nhãn')
        self.ax.set_xlabel('Nhãn thái độ')
        self.ax.set_ylabel('Xác suất')
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = SentimentAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()