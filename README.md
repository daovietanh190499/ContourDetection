## Mã nguồn cài đặt cho hai phương pháp CEDN (Convolutional Encoder Decoder Network) và RCN (RefineContourNet)

  - CEDN: [https://arxiv.org/pdf/1603.04530.pdf](https://arxiv.org/pdf/1603.04530.pdf)
  - RCN: [https://arxiv.org/pdf/1904.13353.pdf](https://arxiv.org/pdf/1904.13353.pdf)

## Huấn luyện với tập dữ liệu Pascal VOC + Pascal Context

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
    name="custom_model"
    lr=1e-4, 
    batch_size=64,
    start_epoch=30,
    max_epoch = 100,
    save_epoch_freq = 1,
    save_iter_freq = 50,
    num_workers = 5,
    loss_func = None,
    images_path="",
    ctns_path="",
    train_path="",
    val_path="",
    model_save_path="",
    model_save_name=""
)
```

Trong đó:
 - `name`: tên cài đặt (bắt buộc, mặc định là `custom_model`)
 - `lr`: tốc độ học (mặc định 1e-4)
 - `batch_size`: kích thước lô (mặc định 64)
 - `start_epoch`: vị trí lần lặp khởi đầu
 - `max_epoch`: vị trí lần lặp tối đa
 - `save_epoch_freq`: tần suất lưu mô hình sau một số lần duyệt (mặc định 1 lần duyệt)
 - `save_iter_freq`: tần suất lưu mô hình sau một số vòng lặp (mặc định 50 vòng lặp)
 - `num_workers`: số lượng workers được khởi tạo cho việc chạy đa tiến trình (mặc định là 5)
 - `loss_func`: hàm thất thoát (mặc định là BCE có trọng số dành cho CEDN) xem cách định nghĩa hàm thất thoát ở dưới
 - `images_path` là đường dẫn tới thư mục ảnh gốc
 - `ctns_path` là đường dẫn tới thư mục nhãn
 - `train_path` là đường dẫn tới tệp `train.txt`
 - `val_path` là đường dẫn tới tệp `val.txt`
 - `model_save_path` là vị trí lưu mô hình sau một số lượng `save_epoch_freq` lần duyệt
 - `model_save_name` là tên mô hình tiền huấn luyện được truyền vào để tiếp tục huấn luyện, nếu huấn luyện từ đầu thì `model_save_name` bằng rỗng

Một số thông tin khác 
 - `optimizer`: Adam
 - `critertion`: BCE

Định nghĩa hàm thất thoát

```
def loss_func(outputs, target):
  thực hiện tính loss ...
```

Trong đó `outputs` là đầu ra của mạng, `target` là nhãn

Hàm loss dành cho CEDN

```
def cedn_loss(outputs, targets):
  weights = torch.empty_like(targets).to(cedn_trainer.device)
  weights[targets >= .97] = 10
  weights[targets < .97] = 1
  res_loss = F.binary_cross_entropy(outputs, targets, weights)
  return res_loss
```

Hàm loss dành cho RCN

```
def rcn_loss(outputs, targets):
  weights = torch.empty_like(targets).to(rcn_trainer.device)
  weights[targets >= .97] = 10
  weights[targets < .97] = 1
  outputs = F.interpolate(outputs, size=(224,224), mode="bilinear", align_corners=False)
  outputs = outputs.to(rcn_trainer.device)
  targets = targets.to(rcn_trainer.device)
  loss = F.binary_cross_entropy(outputs, targets, weights)
  return loss
```

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

## Chạy thử và đánh giá mô hình

Tiến hành chạy thử bằng cách import hàm `eval` từ tệp `eval.py`

```
from eval import eval
result, img = eval(CEDN(), model_path, image_path)
```

Trong đó 
 - `model_path`: đường dẫn tới file pretrain của mô hình
 - `image_path`: đường dẫn tới ảnh
Kết quả trả về gồm 
 - `result`: kết quả dưới dạng ma trận tensor
 - `img`: ảnh gốc dưới dạng ma trận numpy
