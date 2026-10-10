import numpy as np

class GasWellPipelineModel:
    """
    Физическая модель промыслового газопровода (шлейфа)
    на основе уравнения Веймута и дифференциальной динамики релаксации давления.
    """
    def __init__(
        self,
        pipe_length_km: float = 4.5,      # Длина шлейфа: 4.5 км
        pipe_diameter_m: float = 0.203,    # Внутренний диаметр трубы: 203 мм (труба 219х8)
        gas_relative_density: float = 0.60,# Относительная плотность газа (метан ~0.6)
        gas_temperature_k: float = 288.15, # Температура газа (+15 C = 288.15 K)
        z_factor: float = 0.88,            # Коэффициент сверхсжимаемости газа
    ):
        self.L = pipe_length_km
        self.d = pipe_diameter_m
        self.rho_rel = gas_relative_density
        self.T = gas_temperature_k
        self.Z = z_factor

        # Константа сопротивления шлейфа по уравнению Веймута (в СИ)
        # Связывает перепад квадратов давлений и дебит
        # Delta(P^2) = K_friction * Q^2
        self.K_friction = (self.rho_rel * self.T * self.L * self.Z) / (self.d ** (16.0 / 3.0) * 1e8)

        # Постоянная времени инерционности объема газопровода (сек)
        self.tau = 6.0

    def compute_steady_state_pressure(self, p_wellhead_atm: float, flow_rate_m3_day: float) -> float:
        """
        Уравнение Веймута: расчет установившегося давления на входе УКПГ
        по давлению скважины и дебиту.
        """
        # Преобразуем атмосферы в условные единицы давления
        p1_sq = p_wellhead_atm ** 2
        # Падение напора из-за трения о стенки трубы
        pressure_drop_sq = self.K_friction * ((flow_rate_m3_day / 100_000.0) ** 2)

        p2_sq = max(1.0, p1_sq - pressure_drop_sq)
        return float(np.sqrt(p2_sq))

    def simulate(
        self,
        total_seconds: int = 1000,
        dt: float = 0.1,
        seed: int = 42
    ):
        """
        Имитационное моделирование работы шлейфа во времени с реальными возмущениями:
        - Суточные колебания отбора
        - Резкое регулирование штуцера (аварийный сброс / перекрытие)
        - Белый шум АЦП датчика
        - Одиночные сбои измерительного тракта (глитчи 0.1 с)
        """
        np.random.seed(seed)
        n_steps = int(total_seconds / dt)
        t = np.arange(n_steps) * dt

        # 1. Базовые режимы скважины (устьевое давление ~ 75 атм)
        # Медленное истощение пласта + легкие суточные флуктуации пластового давления
        p_wellhead = 75.0 - (0.001 * t) + 0.3 * np.sin(2 * np.pi * t / 600.0)

        # 2. Дебит скважины Q (м3/сут). Базовый дебит = 350 000 м3/сут
        flow_rate = np.full(n_steps, 350_000.0)

        # --- ТЕХНОЛОГИЧЕСКИЕ ВОЗМУЩЕНИЯ (РЕАЛЬНЫЕ СОБЫТИЯ) ---
        # Событие 1: на t=200 сек заклинило регулирующий клапан, расход упал на 5 сек
        idx_event1 = int(200.0 / dt)
        flow_rate[idx_event1 : idx_event1 + int(5.0 / dt)] -= 120_000.0

        # Событие 2: на t=600 сек резкое открытие штуцера на 8 секунд (продувка)
        idx_event2 = int(600.0 / dt)
        flow_rate[idx_event2 : idx_event2 + int(8.0 / dt)] += 150_000.0

        # 3. Интегрирование дифференциального уравнения газодинамики (Релаксация давления)
        p_true_physics = np.zeros(n_steps)
        current_p = self.compute_steady_state_pressure(p_wellhead[0], flow_rate[0])
        p_true_physics[0] = current_p

        alpha_dyn = dt / (self.tau + dt) # Дискретизация уравнения первого порядка

        for i in range(1, n_steps):
            p_target = self.compute_steady_state_pressure(p_wellhead[i], flow_rate[i])
            current_p = current_p + alpha_dyn * (p_target - current_p)
            p_true_physics[i] = current_p

        # 4. Моделирование датчика давления (Sensor Model)
        # Высокочастотный тепловой шум тензопреобразователя датчика
        sensor_noise = np.random.normal(0.0, 0.18, size=n_steps)

        # Аппаратные сбои (одиночные выбросы на 0.1 сек = 1 тик)
        glitch_mask = np.zeros(n_steps)
        glitch_times = [150.0, 202.5, 450.0, 603.2, 820.0]
        for gt in glitch_times:
            idx = int(gt / dt)
            if idx < n_steps:
                glitch_mask[idx] = np.random.uniform(5.0, 8.0) * np.random.choice([-1, 1])

        # Итоговые показания телеметрии, поступающие на вход Rust-ядра
        raw_telemetry = p_true_physics + sensor_noise + glitch_mask

        return t, raw_telemetry, p_true_physics, flow_rate
