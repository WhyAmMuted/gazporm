import numpy as np

class GasPadClusterModel:
    """
    Модель кустовой площадки из 4 скважин и общего манифольда.
    Включает:
    - Дросселирование на штуцере
    - Эффект Джоуля-Томсона (охлаждение при расширении газа)
    - Проверку условий гидратообразования
    - Взаимовлияние скважин через давление коллектора
    """
    def __init__(self):
        # Конфигурация 4 скважин куста
        self.wells_config = {
            "Скважина 101": {"p_wh_base": 88.0, "q_base": 420.0, "t_res": 28.0, "type": "Высокодебитная"},
            "Скважина 102": {"p_wh_base": 72.0, "q_base": 190.0, "t_res": 18.0, "type": "Гидратоопасная"},
            "Скважина 103": {"p_wh_base": 80.0, "q_base": 310.0, "t_res": 24.0, "type": "С помехами КИПиА"},
            "Скважина 104": {"p_wh_base": 76.0, "q_base": 280.0, "t_res": 22.0, "type": "Базовая стабильная"},
        }
        # Коэффициент Джоуля-Томсона для природного газа (град С на 1 атм перепада)
        self.mu_jt = 0.38

    def hydrate_temperature_limit(self, pressure_atm: float) -> float:
        """
        Эмпирическая кривая равновесного гидратообразования для метана.
        Если T_газа <= T_hydrate, начинается образование гидратной пробки!
        """
        p_clamped = max(10.0, pressure_atm)
        # Формула равновесия гидратов природного газа
        return float(8.9 * np.log(p_clamped) - 22.5)

    def simulate_cluster(self, total_seconds: int = 1000, dt: float = 0.1, seed: int = 42):
        np.random.seed(seed)
        n_steps = int(total_seconds / dt)
        t = np.arange(n_steps) * dt

        # Базовое давление в общем коллекторе куста (~55 атм)
        p_collector_base = 55.0 + 0.5 * np.sin(2 * np.pi * t / 400.0)

        telemetry_dataset = {}
        total_cluster_flow = np.zeros(n_steps)

        for well_name, cfg in self.wells_config.items():
            # 1. Устьевое давление P_wh (до штуцера)
            p_wh = cfg["p_wh_base"] - 0.0008 * t + 0.3 * np.sin(2 * np.pi * t / 500.0)

            # 2. Дебит скважины Q (тыс. м3/сут)
            q_flow = np.full(n_steps, cfg["q_base"])

            # Вносим специфические технологические сценарии для скважин
            if well_name == "Скважина 101":
                # Плановое прикрытие штуцера на t=300..340 сек
                mask = (t >= 300) & (t <= 340)
                q_flow[mask] -= 140.0
                p_wh[mask] += 3.5 # Давление перед закрытым штуцером растет

            elif well_name == "Скважина 102":
                # Резкий скачок перепада давления на t=500..560 сек -> риск гидратообразования
                mask = (t >= 500) & (t <= 560)
                q_flow[mask] -= 90.0

            # 3. Линейное давление после штуцера (P_lin) с учетом падения на штуцере
            # Чем больше расход, тем выше давление на входе в коллектор
            p_lin = p_collector_base + (q_flow / 250.0) * 4.0

            # 4. Температура газа после штуцера с эффектом Джоуля-Томсона
            delta_p = np.maximum(0.0, p_wh - p_lin)
            t_gas = cfg["t_res"] - (self.mu_jt * delta_p)

            # 5. Моделирование шума и аппаратных сбоев датчиков КИПиА
            noise_p = np.random.normal(0.0, 0.20, n_steps)
            noise_t = np.random.normal(0.0, 0.12, n_steps)

            p_wh_raw = p_wh + noise_p
            p_lin_raw = p_lin + noise_p
            t_gas_raw = t_gas + noise_t

            # Для Скважины 103 добавляем импульсные наводки датчика (0.1 сек)
            if well_name == "Скважина 103":
                for glitch_t in [150.0, 420.0, 710.0]:
                    idx = int(glitch_t / dt)
                    if idx < n_steps:
                        p_wh_raw[idx] += np.random.choice([-8.0, 8.0])

            # Расчет порога гидратообразования для текущего давления
            t_hydrate_threshold = np.array([self.hydrate_temperature_limit(p) for p in p_lin])
            is_hydrate_risk = t_gas <= (t_hydrate_threshold + 1.5) # зона внимания +1.5 C

            telemetry_dataset[well_name] = {
                "t": t,
                "p_wh_raw": p_wh_raw,
                "p_wh_true": p_wh,
                "p_lin_raw": p_lin_raw,
                "p_lin_true": p_lin,
                "t_gas_raw": t_gas_raw,
                "t_gas_true": t_gas,
                "q_flow": q_flow,
                "t_hydrate_lim": t_hydrate_threshold,
                "hydrate_risk": is_hydrate_risk,
                "type": cfg["type"]
            }
            total_cluster_flow += q_flow

        # Общие параметры манифольда куста
        telemetry_dataset["Манифольд куста"] = {
            "t": t,
            "p_collector_raw": p_collector_base + np.random.normal(0.0, 0.15, n_steps),
            "p_collector_true": p_collector_base,
            "q_total": total_cluster_flow
        }

        return telemetry_dataset
