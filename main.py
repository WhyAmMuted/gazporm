import time
import numpy as np
import matplotlib.pyplot as plt

# Импортируем нашу физическую модель и Rust-модуль
from gas_physics_model import GasWellPipelineModel
import gas_telemetry_edge

print("🚀 Запуск физико-математической имитации скважины и шлейфа...")

# 1. Задаем параметры шлейфа и запускаем симуляцию
model = GasWellPipelineModel(
    pipe_length_km=4.5,       # 4.5 км длина шлейфа
    pipe_diameter_m=0.203,     # диаметр 203 мм
    gas_relative_density=0.60  # метан
)

# Генерируем 100 000 точек (10 000 секунд технологического процесса, частота 10 Гц)
t, raw_sensor, true_physics, flow_rate = model.simulate(total_seconds=10000, dt=0.1)

print(f"📊 Сгенерировано измерений: {len(raw_sensor)}")
print(f"   Базовое давление в шлейфе: ~{true_physics[0]:.2f} атм")

# 2. Передаем честные физические данные в наше Rust-ядро!
start = time.perf_counter()
comp_t, comp_p = gas_telemetry_edge.process_batch(
    t.tolist(),
    raw_sensor.tolist(),
    ema_alpha=0.25,
    sdt_dev_limit=0.30
)
elapsed = time.perf_counter() - start

# 3. Метрики эффективности
orig_len = len(raw_sensor)
comp_len = len(comp_p)
ratio = orig_len / comp_len

print("\n" + "=" * 50)
print(f"⚡ Время обработки ядром Rust: {elapsed:.3f} сек")
print(f"📉 Исходный объем:             {orig_len} точек")
print(f"📦 Сжатый объем:               {comp_len} точек")
print(f"🔥 Коэффициент сжатия:         {ratio:.1f}x (Экономия: {(1 - comp_len/orig_len)*100:.2f}%)")
print("=" * 50)

# 4. Графики для дипломной работы (3 панели)
fig, (ax_flow, ax_full, ax_zoom) = plt.subplots(3, 1, figsize=(14, 11))

# Панель 1: Технологический дебит газа (причина изменений)
ax_flow.plot(t, flow_rate / 1000.0, color='darkgreen', linewidth=1.5)
ax_flow.set_title('Технологический режим: Суточный дебит скважины Q (тыс. м³/сут)', fontweight='bold')
ax_flow.set_ylabel('Q (тыс. м³/сут)')
ax_flow.grid(True, linestyle='--', alpha=0.5)

# Панель 2: Общий обзор давления (Сырое vs Сжатое)
ax_full.plot(t, raw_sensor, color='lightsteelblue', alpha=0.6, label='Сырая телеметрия (Веймут + Шум + Глитчи)')
ax_full.plot(comp_t, comp_p, 'r.-', markersize=3, linewidth=1.2, label=f'Сжатые данные Rust Edge ({comp_len} точек)')
ax_full.set_title(f'Динамика давления в шлейфе по уравнению Веймута (Сжатие в {ratio:.1f} раз)', fontweight='bold')
ax_full.set_ylabel('Давление (атм)')
ax_full.legend(loc='upper right')
ax_full.grid(True, linestyle='--', alpha=0.5)

# Панель 3: Zoom на переходный процесс (срабатывание штуцера на 200 сек + ошибка датчика на 202.5 сек)
zoom_mask_t = (t >= 185) & (t <= 225)
comp_t_arr = np.array(comp_t)
comp_p_arr = np.array(comp_p)
zoom_mask_comp = (comp_t_arr >= 185) & (comp_t_arr <= 225)

ax_zoom.plot(t[zoom_mask_t], raw_sensor[zoom_mask_t], color='blue', alpha=0.3, label='Сырой сигнал датчика')
ax_zoom.plot(t[zoom_mask_t], true_physics[zoom_mask_t], 'k--', linewidth=1.5, label='Истинная физика (Веймут)')
ax_zoom.plot(comp_t_arr[zoom_mask_comp], comp_p_arr[zoom_mask_comp], 'ro-', markersize=6, linewidth=2, label='Сжатый поток (Rust SDT)')

ax_zoom.set_title('Детальный переходный процесс: отработка закрытия штуцера (авария) при фильтрации импульсного шума', fontweight='bold')
ax_zoom.set_xlabel('Время технологического процесса (секунды)')
ax_zoom.set_ylabel('Давление (атм)')
ax_zoom.legend(loc='upper right')
ax_zoom.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("physical_pipeline_result.png", dpi=300)
print("📸 Результаты сохранены в файл: physical_pipeline_result.png")
