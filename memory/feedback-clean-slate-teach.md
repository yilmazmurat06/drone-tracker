---
name: feedback-clean-slate-teach
description: "Tertemiz sayfadan, adım adım, her adımda öğreterek ilerle; eski silinen koda bakma"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cf2c3044-9c99-49c9-9ffc-fea607ff4d12
---

Murat projeyi **tamamen sıfırdan** kurmak istiyor. Git geçmişinde "rewrite"
commit'inin sildiği eksiksiz bir implementasyon var ama **ona bakılmayacak**.

**Why:** Amaç kopyalayıp geçmek değil, mimariyi ilk prensiplerden kurarak
öğrenmek. Eski kodun hatalarını da miras almamak istiyor.

**How to apply:**
- Mimari kararları `initial.txt` + araştırma + ilk prensiplerden türet; eski
  `dtrack` kaynaklarını referans alma.
- Tek seferde her şeyi implemente etme — adım adım ilerle, her modülü ayrı ele al.
- Her adımda kısa "Öğrenecekleri" notu ver; kod yorumlarını öğretici tut.
- Büyük/geri-dönüşü zor kararları (veri kaynağı, threading vb.) önce sor/araştır.

İlgili: [[user-murat]], [[project-drone-tracker-decisions]]
