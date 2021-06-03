### Mã nguồn cài đặt cho hai phương pháp CEDN (Convolutional Encoder Decoder Network) và RCN (RefineContourNet)

  - CEDN: [https://arxiv.org/pdf/1603.04530.pdf](https://arxiv.org/pdf/1603.04530.pdf)
  - RCN: [https://arxiv.org/pdf/1904.13353.pdf](https://arxiv.org/pdf/1904.13353.pdf)

### Huấn luyện với tập dữ liệu Pascal VOC + Pascal Context

Tải tập PASCAL VOC (ảnh huấn luyện) và PASCAL CONTEXT (ảnh nhãn) [tại đây](https://drive.google.com/drive/folders/15g2GNEQtr8ip9Tg-P_cTaK-iRKNZEtjQ?usp=sharing)

Cấu trúc thư mục bao gồm:
  - Thư mục `ctns` chứa các ảnh nhãn
  - Thư mục `images` chứa các ảnh gốc (JPEGImages)
  - Tệp `train.txt` chứa mã của các ảnh huấn luyện, gồm 10581 ảnh
  - Tệp `val.txt` chứa mã của các ảnh kiểm thử, gồm 1449 ảnh

Tiến hành khởi tạo đối tượng Trainer

```
from RCN import rf101
from CEDN import CEDN
from train import Trainer

trainer = Trainer(CEDN()) #Dành cho huấn luyện mô hình CEDN
#trainer = Trainer(rf101()) #Dành cho việc huấn luyện mô hình RCN
```

Thay đổi các cài đặt bằng lệnh `set_config`, bắt buộc phải chạy trong lần đầu huấn luyện

```
trainer.set_config(
    lr=1e-4, 
    batch_size=64,
    start_epoch=30,
    max_epoch = 100,
    num_workers = 5,
    images_path="",
    ctns_path="",
    train_path="",
    val_path="",
    model_save_path="",
    model_save_name=""
)
```

Trong đó:
 - `lr`: tốc độ học (mặc định 1e-4)
 - `batch_size`: kích thước lô (mặc định 64)
 - `start_epoch`: vị trí lần lặp khởi đầu
 - `max_epoch`: vị trí lần lặp tối đa
 - `num_workers`: số lượng workers được khởi tạo cho việc chạy đa tiến trình (mặc định là 5)
 - `images_path` là đường dẫn tới thư mục ảnh gốc
 - `ctns_path` là đường dẫn tới thư mục nhãn
 - `train_path` là đường dẫn tới tệp `train.txt`
 - `val_path` là đường dẫn tới tệp `val.txt`
 - `model_save_path` là vị trí lưu mô hình sau mỗi epoch
 - `model_save_name` là tên mô hình tiền huấn luyện được truyền vào để tiếp tục huấn luyện, nếu huấn luyện từ đầu thì `model_save_name` bằng rỗng

Một số thông tin khác 
 - `optimizer`: Adam
 - `critertion`: BCE

Để tiến hành huấn luyện bằng tối tượng `trainer`

```
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

Đối với các loại dữ liệu nâng cao tự tạo có thể thay đổi đối tượng CustomDataset cho phù hợp

Trong trường hợp muốn sử dụng tệp `train.py` cho mô hình khác chỉ cần truyền mô hình vào đối tượng `Trainer`

Khi sử dụng mã nguồn, đề nghị ghi rõ tên nguồn và tác giả


