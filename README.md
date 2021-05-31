### Huấn luyện với tập dữ liệu Pascal VOC + Pascal Context

Tải tập PASCAL VOC (ảnh huấn luyện) và PASCAL CONTEXT (ảnh nhãn) tại đây 

`
https://drive.google.com/drive/folders/15g2GNEQtr8ip9Tg-P_cTaK-iRKNZEtjQ?usp=sharing
`

Cấu trúc thư mục bao gồm:
  - Thư mục `ctns` chứa các ảnh nhãn
  - Thư mục `images` chứa các ảnh gốc (JPEGImages)
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

Trong đó:
 - `images_path` là đường dẫn tới thư mục ảnh gốc
 - `ctns_path` là đường dẫn tới thư mục nhãn
 - `train_path` là đường dẫn tới tệp `train.txt`
 - `val_path` là đường dẫn tới tệp `val.txt`
 - `model_save_path` là vị trí lưu mô hình sau mỗi epoch
 - `model_save_name` là tên mô hình tiền huấn luyện được truyền vào để tiếp tục huấn luyện, nếu huấn luyện từ đầu thì `model_save_name` bằng rỗng

Một số cài đặt khác
 - `batch_size`: kích thước lô (mặc định 64)
 - `lr`: tốc độ học (mặc định 1e-4)
 - `optimizer`: Adam
 - `critertion`: BCE

Tiến hành huấn luyện bằng đối tượng Trainer

```
from RCN import rf101
from CEDN import CEDN
trainer = Trainer(CEDN()) #Dành cho huấn luyện mô hình CEDN
#trainer = Trainer(rf101()) #Dành cho việc huấn luyện mô hình RCN
trainer.train()
```

Để hiển thị 1 lô trong dataloader

```
trainer.show_dataloader()
```

Để hiển thị biểu đồ thất thoát

```
trainer.loss_plot()
```

Đối với các loại dữ liệu nâng cao tự tạo có thể thay đổi đối tượng Dataloader cho phù hợp
