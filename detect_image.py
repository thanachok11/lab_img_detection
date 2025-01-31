import torch
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ตรวจสอบว่า MPS (GPU บน Mac) ใช้ได้หรือไม่
device = "mps" if torch.backends.mps.is_available() else "cpu"

# โหลดโมเดล YOLOv5 และย้ายไปยังอุปกรณ์ที่พร้อมใช้งาน
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection with YOLOv5")
        self.root.geometry("900x650")
        self.root.configure(bg="#fafafa")
        
        # ตั้งค่าฟอนต์
        self.font = ("Helvetica", 14)
        
        # ส่วนหัว
        self.header_frame = tk.Frame(root, bg="#1E1E1E", pady=20)
        self.header_frame.pack(fill="x")

        self.header_label = tk.Label(self.header_frame, text="Object Detection with YOLOv5", font=("Helvetica", 18), fg="white", bg="#1E1E1E")
        self.header_label.pack()

        # กรอบสำหรับการอัปโหลดภาพและการตรวจจับ
        self.content_frame = tk.Frame(root, bg="#fafafa")
        self.content_frame.pack(pady=30)

        # ปุ่ม Browse
        self.btn_browse = tk.Button(self.content_frame, text="Browse Image", command=self.load_image, font=self.font, bg="#4CAF50", fg="black", relief="flat", width=18, height=2)
        self.btn_browse.grid(row=0, column=0, padx=10)

        # ปุ่ม Detection
        self.btn_detect = tk.Button(self.content_frame, text="Detect Objects", command=self.detect_objects, font=self.font, bg="#2196F3", fg="black", relief="flat", width=18, height=2)
        self.btn_detect.grid(row=0, column=1, padx=10)
        self.btn_detect["state"] = "disabled"  # ปิดปุ่มจนกว่าจะอัปโหลดภาพ
        
        # พื้นที่แสดงภาพ
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(pady=10)

        # สถานะของการตรวจจับ
        self.status_label = tk.Label(root, text="Upload an image to start", font=("Helvetica", 12), bg="#fafafa", fg="#757575")
        self.status_label.pack(pady=10)

        self.image_path = None
        self.img_width, self.img_height = 0, 0

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.btn_detect["state"] = "normal"  # เปิดปุ่ม Detection
            self.status_label.config(text="Image loaded! Click 'Detect Objects' to proceed.", fg="black")

    def display_image(self, path):
        # โหลดภาพ
        img = Image.open(path)
        
        # ปรับขนาดภาพให้พอดีกับ canvas
        self.img_width, self.img_height = img.size
        canvas_width = 600
        canvas_height = 400

        # คำนวณอัตราส่วนเพื่อปรับขนาดภาพให้พอดีกับ canvas
        aspect_ratio = self.img_width / self.img_height
        if self.img_width > self.img_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        
        # ล้าง canvas ก่อนแสดงภาพใหม่
        self.canvas.delete("all")
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width // 2, new_height // 2, image=self.photo)

    def detect_objects(self):
        if self.image_path is None:
            return

        # โหลดภาพและแปลงเป็น NumPy array
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ตรวจจับวัตถุด้วย YOLO
        results = model(img_rgb)

        # ดึงข้อมูลผลลัพธ์ในรูปแบบ pandas DataFrame
        df = results.pandas().xyxy[0]

        # เก็บข้อมูลที่ตรวจจับได้ในข้อความ
        detected_objects = []
        for i, row in df.iterrows():
            label = row['name']
            conf = row['confidence'] * 100  # เปลี่ยนเป็นเปอร์เซ็นต์
            detected_objects.append(f"{label} ({conf:.2f}%)")
            
            # วาดกรอบสี่เหลี่ยมและป้ายชื่อ
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{label} {conf:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ถ้ามีวัตถุที่ตรวจพบ, แสดงข้อความใน status_label
        if detected_objects:
            self.status_label.config(text="Detection complete! Objects detected: " + ", ".join(detected_objects), fg="green")
        else:
            self.status_label.config(text="No objects detected.", fg="red")

        # ปรับขนาดภาพที่จะแสดงใน canvas
        img_width, img_height = img.shape[1], img.shape[0]
        canvas_width = 600
        canvas_height = 400

        # คำนวณอัตราส่วนเพื่อปรับขนาดภาพให้พอดีกับ canvas
        aspect_ratio = img_width / img_height
        if img_width > img_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)

        img_rgb = cv2.resize(img_rgb, (new_width, new_height))

        # แปลงภาพเป็นฟอร์แมตที่ Tkinter รองรับ
        img = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(img)

        # ล้าง canvas ก่อนแสดงภาพใหม่
        self.canvas.delete("all")
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width // 2, new_height // 2, image=self.photo)

    def change_button_color_on_click(self, event):
        # เปลี่ยนสีข้อความเมื่อคลิก
        event.widget.config(fg="black", relief="flat")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
