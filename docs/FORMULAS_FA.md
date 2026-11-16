# فرمول‌های اصلی پیاده‌سازی Full

## DAR

```text
k(e) = 3   for episodes 1–1500
k(e) = 2   for episodes 1501–3000
k(e) = 1   for episodes 3001–6000
```

هر Tick شبیه‌ساز ۰٫۱ ثانیه است؛ DAR فقط تعداد Tickهای اجرای یک اکشن را تغییر می‌دهد.

## ARRS

```text
r_speed = 25 - |v - 50| / 2
heading improvement       = +0.6 × current heading error
heading deterioration     = -6.0 × current heading error
lateral improvement       = +7.2 × current lateral error
lateral deterioration     = -72.0 × current lateral error
```

آستانه ناحیه هم‌راستا: خطای جانبی ≤ ۰٫۵۵ متر و خطای زاویه ≤ ۵ درجه. پاداش عادی به بازه [−۱۰۰، +۳۰] محدود می‌شود؛ جریمه‌های terminal جداگانه‌اند.

## فاصله ثابت اپیزود در نسخه Fixed-Distance

```text
simulator_step = 0.1 s
max_episode_ticks = 1500
simulated_interaction_time = 1500 × 0.1 = 150 s
reference_speed = 50 km/h = 50 / 3.6 m/s
fixed_target_distance = 150 × (50 / 3.6) = 2083.333 m
success_distance = 0.95 × 2083.333 = 1979.167 m
```

سرعت ۵۰ km/h سرعت مرجع Reward است و سرعت خودرو را به‌صورت سخت کنترل نمی‌کند.

```text
route_completion = min(1, cumulative_positive_longitudinal_road_distance / fixed_target_distance)
```

پیشروی از تصویر جابه‌جایی روی میانگین جهت Waypoint لاین رانندگی قبلی و فعلی محاسبه می‌شود. مؤلفه جانبی و حرکت معکوس به مسافت پیشروی اضافه نمی‌شوند.

## Double DQN and Huber

```text
a_next = argmax Q_online(s_next, a) over stored admissible actions
y = R^(k) + (1 - done) × gamma^k × Q_target(s_next, a_next)
Loss = mean_i [w_i × Huber(y_i - Q_online(s_i, a_i))]
```

ماسک اکشن مجازِ ذخیره‌شده در هر Transition برای محاسبه اکشن بعدی استفاده می‌شود.
