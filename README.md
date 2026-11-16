# WILDLIFE-AI

Python tabanlı, modüler vahşi hayvan görüntüleme ve tespit sistemi. Geliştirme
macOS üzerinde yapılır; donanım bağımlılıkları modüller halinde ayrılır ve
üretim hedefi 64-bit Raspberry Pi OS'tur.

## Geliştirme ortamı (Apple Silicon macOS)

Bu çalışma alanında tespit edilen ortam:

- macOS 27.2, Apple Silicon (`arm64`)
- Python 3.13.3 (`arm64`), `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`
- Homebrew kurulu değil; temel kurulum için gerekli değil.

Sanal ortam ve temel bağımlılıklar:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Ortamı doğrulamak için:

```bash
python --version
python -c "import cv2, flask, numpy; print('OpenCV', cv2.__version__); print('Flask', flask.__version__ if hasattr(flask, '__version__') else 'installed'); print('NumPy', numpy.__version__)"
python -m pytest --version
```

YOLO/PyTorch bağımlılıkları temel kurulumdan ayrı tutulur. Bu ayrım, temel
geliştirme kurulumunu hafif tutar ve Apple Silicon MPS ile Raspberry Pi ARM
kurulum seçeneklerinin ayrı doğrulanmasını sağlar:

```bash
python -m pip install -r requirements-vision.txt
```

YOLO bağımlılıkları (Aşama 3):

```bash
python -m pip install -r requirements-vision.txt
python main.py --source video --video /path/to/video.mp4 --detect
```

`--detect` olmadan kamera yalnızca görüntülenir. İlk tespit çalıştırmasında
`yolo11n.pt` ağırlığı gerekirse Ultralytics tarafından indirilir. Alternatif
eğitilmiş modeli `--model /path/to/model.pt` ile seçin. Çizilen etiket ve
`supported_classes` yalnızca modelin kendi sınıf listesini gösterir; COCO
YOLO11n modelinde fox, coyote ve wild boar sınıfları bulunmaz. Tespit aşaması
ses tetiklemez.

## Raspberry Pi hedefi

64-bit Raspberry Pi OS kullanın. Yazılım uyumluluğunu artırmak için Python
3.11 veya 3.12 önerilir; YOLO/PyTorch paketlerinin seçilen Pi modeli, işletim
sistemi ve Python sürümü için uyumlu kurulumu tespit aşamasında ayrıca
doğrulanmalıdır. ESP32-CAM görüntüsü ağ üzerinden alınır; model Raspberry Pi
üzerinde çalışır. Kamera IP/URL'si `config.py` içinden değiştirilir.

## Güvenli başlangıç varsayılanları

- Otomatik caydırma varsayılan olarak kapalıdır.
- Hedef listesi `config.py` içinde kurt, tilki, çakal ve yaban domuzu olarak
  tanımlıdır; evcil köpek otomatik caydırma hedefi değildir.
  Başlangıç YOLO11n/COCO modelinde kurt/tilki/çakal/yaban domuzu sınıfları
  bulunmaz. Model yüklenirken desteklenmeyen hedefler uyarı olarak listelenir;
  bunlar tanınıyormuş gibi yeniden adlandırılmaz.
- COCO ön-eğitimli YOLO sınıflarında tilki, çakal ve yaban domuzu yoktur.
  Bunlar destekleniyormuş gibi ele alınmamalı; özel model/veri seti gerekir.
- İnsan, kedi ve köpek bu aşamada hedef olarak yapılandırılmamıştır.
- Ses seviyesi üst sınırı ve cooldown merkezi `config.py` ayarlarıdır.

## Kamera önizlemesi (Aşama 2)

`camera/` içindeki kamera sınıfları webcam, video dosyası ve ESP32-CAM
HTTP/MJPEG akışını ortak arayüzle sağlar. Canlı kaynaklarda bağlantı
kesildiğinde hata kaydedilir ve `CAMERA_RECONNECT_SECONDS` aralığıyla yeniden
bağlanma denenir. Yerel video dosyasının sonuna ulaşılması normal bitiş sayılır.

Önizlemeyi başlatma:

```bash
source .venv/bin/activate
python main.py --source webcam
python main.py --source video --video /path/to/video.mp4
python main.py --source esp32
```

ESP32-CAM adresini `config.py` içindeki `ESP32_CAMERA_IP` ve
`ESP32_STREAM_URL` ayarlarından değiştirin. ESP32 kamera yazılımı MJPEG stream
sağlamalıdır. Pencereyi `q` ile kapatın; terminalden `Ctrl+C` ile de
durdurabilirsiniz. Penceresiz/sınırlı okuma testi:

```bash
python main.py --source video --video /path/to/video.mp4 --no-display --max-frames 20
python -m pytest -q
```

macOS webcam erişimi için Terminal/VS Code uygulamasına Sistem Ayarları >
Gizlilik ve Güvenlik > Kamera izni verin. ESP32 bağlantı testi aynı komutla
`--source esp32` seçilerek yapılabilir; bağlı olmayan cihazlarda uygulama
çökmeden yeniden bağlanmayı dener.

## SpeciesNet ile tür sınıflandırma denemesi

SpeciesNet kamera tuzağına yönelik hayvan tespiti ve tür sınıflandırmasını
birleştirir. Büyük ve çakışma ihtimali olan bağımlılıkları ayrı sanal ortamda
kurun:

```bash
python3 -m venv .venv-speciesnet
source .venv-speciesnet/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-speciesnet.txt
python main.py --source video --video testvideo.mp4 --detector speciesnet --analysis-every 10
```

SpeciesNet model ağırlıkları ilk çalıştırmada indirilebilir. Model yerelde
çalışır; kareler bir çevrimiçi servise gönderilmez. `--analysis-every 10`
görüntüyü her karede gösterirken her onuncu kareyi analiz eder. Kutuda tür
güveni ve `det` ile nesne tespit güveni gösterilir. Sınıf güveni eşik altında
kalırsa tür adı yerine `animal (uncertain)` gösterilir. SpeciesNet test
ortamında kurulu değilse `requirements-speciesnet.txt` dosyasını o ortamda
kurun. Tür tahminleri gerçek kamera görüntüleriyle doğrulanana kadar otomatik
caydırma kapalı tutulmalıdır.

SpeciesNet kamerası ayrıca `person` kutusunu gösterir; insan tespitleri tür
sınıflandırmasına sokulmaz ve hiçbir koşulda caydırma hedefi olarak
onaylanmaz.

## Aşama 5: güvenli ses ve cooldown

Ses çalıcı MP3/WAV dosyalarını yerel ses çıkışında oynatır. Test etmek için
önce kendi kısa ve makul seviyedeki ses dosyanızı `sounds/` altına koyup:

```bash
python main.py --test-sound --sound sounds/uyari.wav
```

Otomatik ses varsayılan olarak kapalıdır. Yalnızca test ettiğiniz bir
hedef-sınıf eşleşmesi ve ses dosyasıyla açıkça etkinleştirin:

```bash
python main.py --source video --video testvideo.mp4 --detector speciesnet \
  --analysis-every 10 --enable-deterrence
```

ESP32 için `--source esp32` kullanılır. `config.py` içinde
`AUTOMATIC_DETERRENCE_ENABLED` varsayılanı `False`, cooldown 300 saniye,
ses seviyesi üst sınırı 0.25 ve tür başına ses dosyası eşlemesi
`SPECIES_SOUND_FILES` olarak tanımlıdır. Eşleşme yoksa `DEFAULT_SOUND_PATH`
kullanılır. Ses dosyası yoksa hata loglanır ve otomatik caydırma o çalışma
oturumu için devre dışı bırakılır; sahte başarı üretilmez.

Önizleme penceresinde `x` otomatik caydırmayı devre dışı bırakıp devam eden
sesi hemen durdurur; `q` uygulamayı kapatır ve sesi durdurur. İnsan/evcil
köpek otomatik hedef listesinde değildir; ses yalnızca `AnimalTracker`
tarafından ardışık karelerde onaylanan yapılandırılmış hedeflere gider.

`MAX_AUDIO_VOLUME` pygame seviyesini (0-1 arası yazılım gain'i) sınırlar;
bu, hoparlördeki dB SPL değerini ölçmez veya garanti etmez. Dış hoparlör/
amplifikatörde düşük fiziksel sesle başlayın ve hayvanlara zarar verebilecek
seslerden kaçının. Otomatik tetiklemeyi gerçek görüntü ve donanım testinden
önce açmayın.

## Aşama 6: SQLite ve loglar

Uygulama her analiz karesindeki tespitleri `wildlife_ai.sqlite3` dosyasına
yazar. Tekil kare kayıtları `detections`, takip başına özet olaylar `events`
tablosundadır. İnsan tespitleri de tür sınıflandırması yapılmadan geçmişe
kaydedilir; insan görüntüsü snapshot olarak saklanmaz. Snapshot yalnızca hedef
hayvan üç kare doğrulamasını geçtiğinde `snapshots/` içine yazılır.

Uygulama logları `logs/` altında `wildlife_ai.log`, `camera.log`,
`detection.log`, `deterrent.log` ve `database.log` dosyalarına ayrılır.
Dosyalar uygulama başlarken oluşturulur.

Örnek geçmiş sorgusu:

```bash
sqlite3 wildlife_ai.sqlite3 \
  "SELECT species, confidence, detected_at, target_confirmed FROM detections ORDER BY detected_at DESC LIMIT 20;"
sqlite3 wildlife_ai.sqlite3 \
  "SELECT species, last_seen_at, snapshot_path, audio_triggered, trigger_reason FROM events ORDER BY last_seen_at DESC LIMIT 20;"
```

Bağımlılıklar ve SQLite entegrasyonu test edilmiştir; olay ve tespit kayıtları
uygulamanın aynı süreçte kullandığı veritabanına kalıcı olarak yazılır.

## Aşama 7: web yönetim paneli

Panel aynı kamera/algılama/izleme akışını arka plan iş parçacığında çalıştırır;
ayrı bir kamera oturumu açmaz. Başlatma ve durdurma, canlı MJPEG görüntüsü,
kamera/model durumu, son tespit, caydırma/cooldown bilgisi, olay geçmişi ve
olay snapshot'ları panelden yönetilebilir. Panelin bağımlılığı
`requirements.txt` içinde zaten bulunan Flask'tır.

Önce kamera kaynağını `config.py` içinden seçin. YOLO için temel sanal
ortamınızda `requirements-vision.txt` kurulu olmalıdır; SpeciesNet seçilecekse
`DASHBOARD_DETECTOR = "speciesnet"` yapıp SpeciesNet bağımlılıklarını o Python
ortamına kurun. Paneli başlatın:

```bash
source .venv/bin/activate
python -m dashboard.app
```

Tarayıcıdan `http://127.0.0.1:5000` adresini açın. Varsayılan olarak panel
yalnızca aynı bilgisayardan erişilebilir. Yerel ağdan erişim gerekiyorsa
`config.py` içindeki `DASHBOARD_HOST` değerini `"0.0.0.0"` yapın ve sunucuyu
başlatmadan önce güçlü bir parola belirleyin:

```bash
export WILDLIFE_DASHBOARD_PASSWORD='uzun-ve-benzersiz-bir-parola'
python -m dashboard.app
```

LAN erişiminde tarayıcı HTTP Basic Auth sorar; kullanıcı adı serbesttir,
parola yalnızca ortam değişkeninden okunur. Uzak adrese bağlanırken parola
yoksa panel başlamaz. Düz HTTP kullanımı yalnızca güvenilir yerel ağda uygundur;
güvenilmeyen ağlarda VPN veya TLS korumalı ters vekil kullanın. Parolayı,
video/snapshot'ları, veritabanını ve logları Git'e eklemeyin.

Panelde otomatik caydırma ayrıca ve açıkça etkinleştirilmelidir; varsayılanı
kapalıdır. Manuel ses testi sadece sistem durmuşken kullanılabilir ve
`DEFAULT_SOUND_PATH` dosyasını çalar. Acil durdurma devam eden sesi keser ve
caydırmayı kapatır. Eşik, hedef ve cooldown güvenlik kontrolleri panel
tarafından değiştirilmez; `config.py` ve mevcut takip/tetikleme kuralları
geçerlidir. Panel işlevleri Flask test istemcisi ve mock kamera/ses ile
donanımsız doğrulanabilir:

```bash
python -m pytest -q tests/test_dashboard.py
```

## Proje aşamaları

1. Proje iskeleti, ortam ve konfigürasyon (tamamlandı)
2. Kamera kaynakları: webcam, dosya ve ESP32-CAM (tamamlandı)
3. YOLO tespiti ve görüntü üzerine çizim (tamamlandı; SpeciesNet seçilebilir)
4. Ardışık kare doğrulama ve hedef takibi (tamamlandı)
5. Güvenli ses çalma ve cooldown (tamamlandı; saha testi bekleniyor)
6. SQLite ve kategorili loglama (tamamlandı)
7. Yerel web yönetim paneli (tamamlandı)
8. Testler ve Raspberry Pi uyumluluğu

Her aşama doğrulandıktan sonra bir sonraki aşamaya geçin.

## Aşama 4: hedef doğrulama ve takip

`AnimalTracker`, yalnızca `config.py` içindeki hedef tür adlarını/alias'larını
ve güven eşiğini sağlayan tespitleri hedef adayı sayar. Varsayılan olarak aynı
nesnenin üç **analiz edilen karede** art arda görülmesi gerekir; atlanan video
kareleri doğrulama sayacını artırmaz. Aradaki analiz karesinde nesne
bulunmazsa art arda doğrulama sayacı sıfırlanır. Kısa süreli kayıplarda olay
kimliği takip devamlılığı için korunur.

Eşleşme aynı sınıf ve bounding-box IoU ile yapılır. `approaching`,
`moving_away` ve `stable` yalnız sabit kamerada kutu alanı trendinden
türetilmiş yaklaşık durumlardır; kesin fiziksel mesafe ölçümü değildir.
Türü belirsiz tahminler hiçbir zaman hedef olarak onaylanmaz. Onay yalnızca
log/etiket üretir; bu aşamada ses tetikleme yoktur. Ayarlar
`TARGET_ANIMALS`, `CONFIDENCE_THRESHOLD`,
`CONSECUTIVE_CONFIRMATION_FRAMES` ve `TRACK_*` değişkenlerindedir.
