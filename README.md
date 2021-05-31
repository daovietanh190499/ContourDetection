### Huấn luyện với tập dữ liệu Pascal VOC + Pascal Context

Cấu trúc thư mục bao gồm
  - Thư mục cnts chứa các ảnh nhãn
  - Thư mục images chứa các ảnh gốc (JPEGImages)
  - Tệp `train.txt` chứa mã của các ảnh huấn luyện, gồm 10581 ảnh
  - Tệp `val.txt` chứa mã của các ảnh kiểm thử, gồm 1449 ảnh

Thay đổi các tham số trong file train.py

```
self.images_path = "/content/drive/MyDrive/convert_dataset/images/"
self.ctns_path = "/content/drive/MyDrive/convert_dataset/ctns/"
self.train_path = "/content/drive/MyDrive/train.txt"
self.val_path = "/content/drive/MyDrive/val.txt"
self.model_save_path = "/content/drive/MyDrive/model_seg/"
self.model_save_name = "cedn_epoch_9.pth"
```

Trong đó 
 - `images_path` là đường dẫn tới thư mục ảnh gốc
 - `ctns_path` là đường dẫn tới thư mục nhãn
 - `train_path` là đường dẫn tới tệp `train.txt`
 - `val_path` là đường dẫn tới tệp `val.txt`
 - `model_save_path` là vị trí lưu mô hình sau mỗi epoch
 - `model_save_name` là tên mô hình tiền huấn luyện được truyền vào để tiếp tục huấn luyện, nếu huấn luyện từ đầu thì `model_save_name` bằng rỗng

Huấn luyện bằng đối tượng Trainer

```
from RCN import rf101
from CEDN import CEDN
trainer = Trainer(CEDN())
#trainer = Trainer(rf101())
trainer.train()
```
