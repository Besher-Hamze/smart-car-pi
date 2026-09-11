# مسار مطبوع: طريق رمادي + إطار أصفر + خط أبيض متقطع بالنص.

## تشغيل تلقائي مع الإقلاع
على الراسبيري مرة واحدة:
```bash
chmod +x ~/smart-car-pi/scripts/start.sh
sudo cp ~/smart-car-pi/scripts/smart-car.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable smart-car
sudo systemctl start smart-car
```
بعدها كل ما تشغّل الراسبيري، الكود بيشتغل لحاله.

إيقاف: `sudo systemctl stop smart-car`
تعطيل الإقلاع: `sudo systemctl disable smart-car`
السجل: `journalctl -u smart-car -f`

## تشغيل يدوي
```bash
sudo systemctl stop smart-car
pkill -f 'python3 main.py'
cd ~/smart-car-pi
source .venv/bin/activate
python3 main.py
```
على اللابتوب افتح: `http://IP-الراسبيري:8080/`

## علّم السيارة 3 لفات (أسلوب Donkeycar / DIY Robocars)
نفس فكرة [diyrobocars.com](https://www.diyrobocars.com/): أنت تسوق، شبكة عصبونية بتنسخ التوجيه من صورة الكاميرا.

1. **MANUAL** ثم **REC** (أحمر).
2. سوق **3 لفات نظيفة** على الخريطة.
3. أوقف REC ثم اضغط **TRAIN** (بنفس الصفحة — بدون ترمينال).
4. انتظر الرسالة تحت (دقايق). بعدين **AUTO** لازم يطلع `PILOT`.

أول مرة على الراسبيري:
```bash
cd ~/smart-car-pi && source .venv/bin/activate
pip install tensorflow scikit-learn joblib
```

الكاميرا تميل للأسفل. النقطة الخضراء لازم تكون على الخط الأبيض بالنص.

- جهة تمشي عكس: `INVERT_LEFT` أو `INVERT_RIGHT` في `config.py`
- تلف عكس الخط: `STEER_SIGN = -1`
- أسرع/أبطأ: `SPEED`

## توصيل L298N
ENA=12  IN1=17  IN2=27  ENB=13  IN3=22  IN4=23
أزل قافز ENA و ENB. GND مشترك. بطارية المحركات منفصلة.
