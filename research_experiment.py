import time
import numpy as np
import matplotlib.pyplot as plt
from gas_physics_model import GasWellPipelineModel
import gas_telemetry_edge

print("🔬 Запуск исследовательского эксперимента (Блок 3: Оценка погрешностей)...")

# 1. Генерируем тестовый технологический процесс (10 000 сек, 100 000 точек)
model = GasWellPipelineModel()
t, raw_sensor, true_physics, _ = model.simulate(total_seconds=10000, dt=0.1, seed=42)

# =====================================================================
# ЭКСПЕРИМЕНТ 1: ОЦЕНКА КАЧЕСТВА ВОССТАНОВЛЕНИЯ ПРИ ТЕКУЩЕМ DEV_LIMIT
# =====================================================================
current_dev = 0.30
comp_t, comp_p = gas_telemetry_edge.process_batch(
    t.tolist(), raw_sensor.tolist(), ema_alpha=0.25, sdt_dev_limit=current_dev
)

# Восстанавливаем сигнал на исходную 10 Гц сетку через линейную интерполяцию
reconstructed_p = np.interp(t, comp_t, comp_p)

# Расчет метрик относительно ИСТИННОЙ ФИЗИКИ (а не зашумленного датчика!)
residual_error = np.abs(true_physics - reconstructed_p)
mae = np.mean(residual_error)
rmse = np.sqrt(np.mean((true_physics - reconstructed_p) ** 2))
prd = (np.sqrt(np.sum((true_physics - reconstructed_p) ** 2)) / np.sqrt(np.sum(true_physics ** 2))) * 100.0
max_err = np.max(residual_error)

print("\n" + "=" * 55)
print(f" МЕТРОЛОГИЧЕСКИЕ ХАРАКТЕРИСТИКИ (E = {current_dev} атм)")
print("=" * 55)
print(f"Средняя абсолютная ошибка (MAE):     {mae:.4f} атм")
print(f"Среднеквадратичная ошибка (RMSE):    {rmse:.4f} атм")
print(f"Относительная ошибка (PRD):          {prd:.3f} %")
print(f"Максимальное расхождение (Max Err):  {max_err:.4f} атм")
print(f"Коэффициент сжатия:                  {len(raw_sensor) / len(comp_p):.1f}x")
print("=" * 55)

# =====================================================================
# ЭКСПЕРИМЕНТ 2: ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ И КРИВАЯ ПАРЕТО
# =====================================================================
print("\n[+] Расчет Парето-кривой компромисса (Trade-off: сжатие vs точность)...")

# Перебираем коридор погрешности от 0.05 до 1.0 атм
dev_limits = np.linspace(0.05, 1.0, 20)
compression_ratios = []
rmse_list = []
max_err_list = []

for dev in dev_limits:
    ct, cp = gas_telemetry_edge.process_batch(
        t.tolist(), raw_sensor.tolist(), ema_alpha=0.25, sdt_dev_limit=float(dev)
    )
    # Интерполируем
    recon = np.interp(t, ct, cp)

    # Метрики
    cr = len(raw_sensor) / len(cp)
    cur_rmse = np.sqrt(np.mean((true_physics - recon) ** 2))
    cur_max = np.max(np.abs(true_physics - recon))

    compression_ratios.append(cr)
    rmse_list.append(cur_rmse)
    max_err_list.append(cur_max)

print("[+] Расчет завершен. Формирование научных графиков...")

# =====================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ИССЛЕДОВАНИЯ
# =====================================================================
fig = plt.figure(figsize=(15, 10))

# График 1: Истинный процесс vs Восстановленный + график ошибки Delta P
ax1 = plt.subplot(2, 2, (1, 2))
ax1.plot(t, true_physics, 'k-', linewidth=1.5, label='Истинный физический процесс (Веймут)')
ax1.plot(t, reconstructed_p, 'r--', linewidth=1.2, label=f'Восстановленный сигнал из {len(comp_p)} точек')
ax1.set_title('Качество восстановления технологического давления во времени', fontweight='bold', fontsize=12)
ax1.set_ylabel('Давление (атм)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')

# Дополнительная шкала снизу для отображения абсолютной ошибки
ax_err = ax1.twinx()
ax_err.plot(t, residual_error, color='gray', alpha=0.35, label='Локальная погрешность |P_true - P_recon|')
ax_err.set_ylabel('Ошибка (атм)', color='gray')
ax_err.set_ylim(0, max_err * 3.0)

# График 2: Кривая Парето (Степень сжатия vs Погрешность RMSE)
ax2 = plt.subplot(2, 2, 3)
ax2.plot(dev_limits, compression_ratios, 'b-o', linewidth=2, markersize=5)
ax2.set_title('Зависимость сжатия от допустимой погрешности E', fontweight='bold')
ax2.set_xlabel('Допустимый коридор E (атм)')
ax2.set_ylabel('Коэффициент сжатия (раз)', color='blue')
ax2.grid(True, linestyle='--', alpha=0.5)

# График 3: Погрешность RMSE vs Коридор E
ax3 = plt.subplot(2, 2, 4)
ax3.plot(dev_limits, rmse_list, 'r-s', linewidth=2, markersize=5, label='RMSE')
ax3.plot(dev_limits, max_err_list, 'orange', linestyle='--', linewidth=1.5, label='Max Error')
ax3.axhline(y=current_dev, color='black', linestyle=':', label='Текущая уставка E')
ax3.set_title('Рост погрешности при загрублении порога E', fontweight='bold')
ax3.set_xlabel('Допустимый коридор E (атм)')
ax3.set_ylabel('Погрешность (атм)')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend()

plt.tight_layout()
plt.savefig("experiment_tradeoff_result.png", dpi=300)
print("📸 Научные графики сохранены в: experiment_tradeoff_result.png")
