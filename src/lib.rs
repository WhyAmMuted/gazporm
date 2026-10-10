use pyo3::prelude::*;

// =====================================================================
// 1. БАЗОВАЯ МАТЕМАТИКА И АИСД (НА СТЕКЕ, БЕЗ АЛЛОКАЦИЙ)
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryPoint {
    pub timestamp: f64,
    pub pressure: f64,
}

pub struct MedianFilter3 {
    window: [f64; 3],
    count: usize,
}

impl MedianFilter3 {
    pub fn new() -> Self {
        Self {
            window: [0.0; 3],
            count: 0,
        }
    }

    pub fn update(&mut self, val: f64) -> f64 {
        if self.count < 3 {
            self.window[self.count] = val;
            self.count += 1;
            return val;
        }
        self.window[0] = self.window[1];
        self.window[1] = self.window[2];
        self.window[2] = val;

        let (a, b, c) = (self.window[0], self.window[1], self.window[2]);
        if (a <= b && b <= c) || (c <= b && b <= a) {
            b
        } else if (b <= a && a <= c) || (c <= a && a <= b) {
            a
        } else {
            c
        }
    }
}

pub struct EmaFilter {
    alpha: f64,
    prev_val: Option<f64>,
}

impl EmaFilter {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            prev_val: None,
        }
    }

    pub fn update(&mut self, val: f64) -> f64 {
        match self.prev_val {
            Some(prev) => {
                let current = self.alpha * val + (1.0 - self.alpha) * prev;
                self.prev_val = Some(current);
                current
            }
            None => {
                self.prev_val = Some(val);
                val
            }
        }
    }
}

pub struct SdtCompressor {
    dev_limit: f64,
    start_point: Option<TelemetryPoint>,
    last_point: Option<TelemetryPoint>,
    max_slope: f64,
    min_slope: f64,
}

impl SdtCompressor {
    pub fn new(dev_limit: f64) -> Self {
        Self {
            dev_limit,
            start_point: None,
            last_point: None,
            max_slope: f64::INFINITY,
            min_slope: f64::NEG_INFINITY,
        }
    }

    pub fn push(&mut self, current: TelemetryPoint) -> Option<TelemetryPoint> {
        let start = match self.start_point {
            None => {
                self.start_point = Some(current);
                self.last_point = Some(current);
                return Some(current);
            }
            Some(s) => s,
        };

        let dt = current.timestamp - start.timestamp;
        if dt <= 0.0 {
            return None;
        }

        let slope_to_upper = (current.pressure + self.dev_limit - start.pressure) / dt;
        let slope_to_lower = (current.pressure - self.dev_limit - start.pressure) / dt;

        if slope_to_upper < self.max_slope {
            self.max_slope = slope_to_upper;
        }
        if slope_to_lower > self.min_slope {
            self.min_slope = slope_to_lower;
        }

        if self.min_slope > self.max_slope {
            let emitted = self.last_point.unwrap();
            self.start_point = Some(emitted);
            let new_dt = current.timestamp - emitted.timestamp;
            self.max_slope = (current.pressure + self.dev_limit - emitted.pressure) / new_dt;
            self.min_slope = (current.pressure - self.dev_limit - emitted.pressure) / new_dt;
            self.last_point = Some(current);
            Some(emitted)
        } else {
            self.last_point = Some(current);
            None
        }
    }

    pub fn flush(&mut self) -> Option<TelemetryPoint> {
        self.last_point.take()
    }
}

pub struct EdgeTelemetryPipeline {
    median: MedianFilter3,
    ema: EmaFilter,
    sdt: SdtCompressor,
}

impl EdgeTelemetryPipeline {
    pub fn new(ema_alpha: f64, sdt_dev_limit: f64) -> Self {
        Self {
            median: MedianFilter3::new(),
            ema: EmaFilter::new(ema_alpha),
            sdt: SdtCompressor::new(sdt_dev_limit),
        }
    }

    #[inline(always)]
    pub fn process_sample(&mut self, raw: TelemetryPoint) -> Option<TelemetryPoint> {
        let med = self.median.update(raw.pressure);
        let clean = self.ema.update(med);
        let pt = TelemetryPoint {
            timestamp: raw.timestamp,
            pressure: clean,
        };
        self.sdt.push(pt)
    }
}

// =====================================================================
// 2. ИНТЕРФЕЙС ЭКСПОРТА В PYTHON (PyO3)
// =====================================================================

/// Потоковый класс для Python
#[pyclass]
pub struct PyEdgePipeline {
    inner: EdgeTelemetryPipeline,
}

#[pymethods]
impl PyEdgePipeline {
    #[new]
    pub fn new(ema_alpha: f64, sdt_dev_limit: f64) -> Self {
        Self {
            inner: EdgeTelemetryPipeline::new(ema_alpha, sdt_dev_limit),
        }
    }

    /// Принимает (t, p), возвращает кортеж (t, p) или None
    pub fn process_sample(&mut self, timestamp: f64, pressure: f64) -> Option<(f64, f64)> {
        let raw = TelemetryPoint {
            timestamp,
            pressure,
        };
        self.inner
            .process_sample(raw)
            .map(|pt| (pt.timestamp, pt.pressure))
    }

    pub fn flush(&mut self) -> Option<(f64, f64)> {
        self.inner.sdt.flush().map(|pt| (pt.timestamp, pt.pressure))
    }
}

/// Высокоскоростная пакетная функция для массивов NumPy
#[pyfunction]
pub fn process_batch(
    times: Vec<f64>,
    pressures: Vec<f64>,
    ema_alpha: f64,
    sdt_dev_limit: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if times.len() != pressures.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Размеры массивов times и pressures должны совпадать",
        ));
    }

    let mut pipeline = EdgeTelemetryPipeline::new(ema_alpha, sdt_dev_limit);
    let mut out_times = Vec::new();
    let mut out_pressures = Vec::new();

    for i in 0..times.len() {
        let raw = TelemetryPoint {
            timestamp: times[i],
            pressure: pressures[i],
        };

        if let Some(saved) = pipeline.process_sample(raw) {
            out_times.push(saved.timestamp);
            out_pressures.push(saved.pressure);
        }
    }

    if let Some(last) = pipeline.sdt.flush() {
        out_times.push(last.timestamp);
        out_pressures.push(last.pressure);
    }

    Ok((out_times, out_pressures))
}

/// Регистрация Python-модуля
#[pymodule]
fn gas_telemetry_edge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEdgePipeline>()?;
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
