# 🍓 WILDLIFE-AI: Raspberry Pi 4 Kurulum Kılavuzu

Bu belge, **WILDLIFE-AI** projesini 64-bit Raspberry Pi 4 üzerinde en yüksek performansla (ONNX motoru ile) sıfırdan kurup çalıştırmanız için gereken tüm adımları ve terminal komutlarını içerir.

---

## 📌 1. AŞAMA: SD Kart ve İşletim Sistemi Hazırlığı

1. Bilgisayarınıza **Raspberry Pi Imager** uygulamasını indirin: [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Programı açın:
   * **Device:** Raspberry Pi 4
   * **Operating System:** `Raspberry Pi OS (64-bit)` *(ÖNEMLİ: Mutlaka 64-bit Debian Bookworm seçin)*
   * **Storage:** En az 16 GB (Tercihen 32 GB Class 10 / A1) MicroSD kartınız
3. **Ayarları Düzenle (Dişli çark simgesi):**
   * Kullanıcı adı (örn: `pi`) ve şifrenizi belirleyin.
   * Ev/araç Wi-Fi bilgilerinizi girin.
   * **SSH'yi etkinleştirin** (Şifre ile giriş).
4. **Yazdır (Write)** butonuna basarak kartı hazırlayın ve Raspberry Pi 4'e takıp çalıştırın.

---

## 📌 2. AŞAMA: Sistem Paketleri ve Sürücülerin Kurulumu

Raspberry Pi'a SSH veya doğrudan klavye/ekran ile bağlanın ve terminali açın:

```bash
# 1. Sistemi güncelleyin
sudo apt update && sudo apt upgrade -y

# 2. Görüntü ve ses için gerekli sistem kütüphanelerini kurun
sudo apt install -y python3-pip python3-venv git libgl1 libglib2.0-0 libatlas-base-dev libportaudio2 libasound2-dev alsa-utils
```

---

## 📌 3. AŞAMA: Projeyi Raspberry Pi'a Klonlama / Kopyalama

Projenizi Raspberry Pi'a iki yoldan biriyle aktarabilirsiniz:

### Seçenek A: Git ile Klonlama (Önerilen)
```bash
cd ~
git clone <GITHUB_REPO_LINKINIZ> WILDLIFE-AI
cd WILDLIFE-AI
```

### Seçenek B: Bilgisayarınızdan SCP ile Kopyalama
Windows PowerShell üzerinden tek komutla aktarabilirsiniz (IP adresini kendi RPi IP'nizle değiştirin):
```powershell
scp -r C:\Users\CAN\Desktop\Projects\WILDLIFE-AI pi@192.168.1.XX:~/WILDLIFE-AI
```

---

## 📌 4. AŞAMA: Python Sanal Ortamı ve Hafif Bağımlılıklar

Raspberry Pi'da gigabaytlarca ağır PyTorch derlemekle uğraşmıyoruz; projemiz için hazırladığımız optimize **ONNX Runtime** paketlerini kuruyoruz:

```bash
cd ~/WILDLIFE-AI

# 1. Sanal ortam oluşturun
python3 -m venv .venv

# 2. Sanal ortamı etkinleştirin
source .venv/bin/activate

# 3. Pip'i güncelleyin
pip install --upgrade pip

# 4. PyTorch'un sadece CPU sürümünü kurun (Gereksiz gigabaytlarca NVIDIA/CUDA paketini engellemek için ŞART):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. RPi için optimize edilmiş hafif paketleri kurun
pip install -r requirements-rpi.txt
```

---

## 📌 5. AŞAMA: Donanım Testleri (Kamera ve Hoparlör)

### A. Ses Çıkışını Test Edin:
3.5mm ses jakına veya USB hoparlöre ses kablosunu bağlayın:
```bash
# Sesi test edin (10 saniyelik siren çalmalıdır)
python main.py --test-sound
```
*(Ses seviyesini ayarlamak için terminalde `alsamixer` komutunu kullanabilirsiniz).*

### B. Kamera Bağlantısını Kontrol Edin:
* **USB Webcam kullanıyorsanız:**
  ```bash
  ls /dev/video*
  # Genellikle /dev/video0 olarak görünür.
  ```
* **ESP32-CAM kullanıyorsanız:**
  `config.py` dosyasındaki `ESP32_CAMERA_IP` değerine ESP32'nin yerel ağ IP'sini yazın.

---

## 📌 6. AŞAMA: Sistemi Çalıştırma

### 1. Terminal ve Kamera ile Test:
```bash
source .venv/bin/activate
python main.py --model models/yolo11n.onnx --detect
```

### 2. Web Kontrol Panelini (Dashboard) Başlatma (Tüm Ağa Açık):
Raspberry Pi monitörsüz (headless) çalışırken aynı Wi-Fi ağındaki telefonunuzdan, tabletinizden veya bilgisayarınızdan canlı izlemek için:

```bash
source .venv/bin/activate

# Flask uygulamasını yerel ağa (0.0.0.0) açın
export FLASK_APP="dashboard.app:create_app()"
export WILDLIFE_DASHBOARD_PASSWORD="guclu_bir_sifre" # Ağ dışından erişim için güvenlik şifresi
flask run --host 0.0.0.0 --port 5000
```

Artık telefonunuzun veya bilgisayarınızın tarayıcısını açıp şu adrese gidin:
👉 **`http://raspberrypi.local:5000`** *(veya `http://192.168.1.XX:5000`)*

---

## 📌 7. AŞAMA: Elektrik Geldiğinde Otomatik Başlatma (Systemd Servisi)

Tarlada veya arazide elektrik kesilip geldiğinde Raspberry Pi'ın ekran/klavye istemeden sistemi arka planda otomatik başlatması için:

1. Yeni bir servis dosyası oluşturun:
   ```bash
   sudo nano /etc/systemd/system/wildlife.service
   ```

2. İçine şu satırları yapıştırın (kullanıcı adınız `pi` ise):
   ```ini
   [Unit]
   Description=WILDLIFE-AI Akıllı Yaban Hayatı Gözlem ve Caydırma Servisi
   After=network.target

   [Service]
   User=pi
   WorkingDirectory=/home/pi/WILDLIFE-AI
   Environment="PATH=/home/pi/WILDLIFE-AI/.venv/bin:/usr/bin"
   Environment="FLASK_APP=dashboard.app:create_app()"
   ExecStart=/home/pi/WILDLIFE-AI/.venv/bin/flask run --host 0.0.0.0 --port 5000
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

3. Servisi etkinleştirin ve başlatın:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable wildlife.service
   sudo systemctl start wildlife.service
   ```

4. Servis durumunu kontrol etmek için:
   ```bash
   sudo systemctl status wildlife.service
   ```

---

## 💡 İpuçları ve Sorun Giderme

* **Sıcaklık Kontrolü:** Sürekli kamera analizi sırasında işlemci sıcaklığını görmek için:
  ```bash
  vcgencmd measure_temp
  ```
  *(75°C üzerindeyse küçük bir fan veya alüminyum soğutucu gövde önerilir).*
* **Daha Yüksek FPS İhtiyacı:** Eğer 640x640 çözünürlükte FPS'i artırmak isterseniz, bilgisayarınızda modeli 320x320 olarak export edip RPi'a aktarın:
  ```powershell
  python tools/export_rpi.py --model yolo11n.pt --format onnx --imgsz 320
  ```
