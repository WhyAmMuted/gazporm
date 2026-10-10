import numpy as np
from scipy.optimize import minimize
from pad_cluster_model import GasPadClusterModel

class GasPadOptimizer:
    """
    Система усовершенствованного управления технологическим процессом (APC).
    Находит оптимальное положение штуцеров скважин куста.
    """
    def __init__(self, model: GasPadClusterModel):
        self.model = model
        self.well_names = list(model.wells_config.keys())
        self.n_wells = len(self.well_names)

    def evaluate_pad(self, choke_positions: np.ndarray):
        """
        choke_positions: массив коэффициентов открытия штуцеров [u_1, u_2, u_3, u_4] от 0.1 до 1.0
        Возвращает: суммарный дебит, давление в коллекторе, запасы по гидратам
        """
        flows = []
        p_wh_list = []

        for idx, w_name in enumerate(self.well_names):
            cfg = self.model.wells_config[w_name]
            u = choke_positions[idx]

            # Дебит зависит от открытия штуцера
            q = cfg["q_base"] * u
            flows.append(q)

            # Устьевое давление: чем сильнее открыт штуцер, тем ниже давление на устье (депрессия)
            p_wh = cfg["p_wh_base"] - (1.0 - u) * 4.0
            p_wh_list.append(p_wh)

        total_q = np.sum(flows)

        # Давление в общем манифольде куста
        p_collector = 55.0 + (total_q - 1000.0) / 150.0 * 2.0

        # Расчет температур и гидратных запасов
        hydrate_margins = []
        t_gas_list = []
        p_lin_list = []

        for idx, w_name in enumerate(self.well_names):
            cfg = self.model.wells_config[w_name]
            q = flows[idx]
            p_wh = p_wh_list[idx]

            p_lin = p_collector + (q / 250.0) * 3.5
            delta_p = max(0.0, p_wh - p_lin)
            t_gas = cfg["t_res"] - (self.model.mu_jt * delta_p)

            t_hyd = self.model.hydrate_temperature_limit(p_lin)
            margin = t_gas - t_hyd

            hydrate_margins.append(margin)
            t_gas_list.append(t_gas)
            p_lin_list.append(p_lin)

        return {
            "total_q": total_q,
            "flows": flows,
            "p_collector": p_collector,
            "p_wh": p_wh_list,
            "p_lin": p_lin_list,
            "t_gas": t_gas_list,
            "hydrate_margins": hydrate_margins
        }

    def optimize_chokes(self, max_p_collector: float = 58.0, min_hydrate_margin: float = 2.0):
        """
        Решает задачу оптимизации:
        max(Total_Q) при P_coll <= max_p_collector и hydrate_margin >= min_hydrate_margin
        """
        # Начальное приближение (все штуцеры открыты на 80%)
        x0 = np.full(self.n_wells, 0.8)
        bounds = [(0.2, 1.0) for _ in range(self.n_wells)]

        # Минимизируем минус дебит (что эквивалентно максимизации добычи)
        def objective(x):
            res = self.evaluate_pad(x)
            return -res["total_q"]

        # Ограничение 1: Давление в коллекторе <= max_p_collector
        def constraint_p_coll(x):
            res = self.evaluate_pad(x)
            return max_p_collector - res["p_collector"]

        # Ограничение 2: Запас по гидратам на каждой скважине >= min_hydrate_margin
        constraints = [
            {'type': 'ineq', 'fun': constraint_p_coll}
        ]

        for i in range(self.n_wells):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, well_idx=i: self.evaluate_pad(x)["hydrate_margins"][well_idx] - min_hydrate_margin
            })

        opt_result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-4, 'maxiter': 100}
        )

        best_x = opt_result.x
        eval_before = self.evaluate_pad(x0)
        eval_after = self.evaluate_pad(best_x)

        return {
            "success": opt_result.success,
            "chokes_before": x0,
            "chokes_after": best_x,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "gain_q": eval_after["total_q"] - eval_before["total_q"],
            "gain_percent": ((eval_after["total_q"] / eval_before["total_q"]) - 1.0) * 100.0
        }
