/**
 * Prediction page JavaScript
 * Handles area selection and renders 7-day forecast charts.
 * Theme-aware: reads CSS variables for dark/light mode chart colors.
 */

var predictBtn = document.getElementById("predict-btn");
var areaSelect = document.getElementById("area-select");

areaSelect.addEventListener("change", function () {
    predictBtn.disabled = !this.value;
});

predictBtn.addEventListener("click", function () {
    var area = areaSelect.value;
    if (!area) return;
    loadPrediction(area);
});

function loadPrediction(area) {
    document.getElementById("prediction-content").style.display = "none";
    document.getElementById("prediction-error").style.display = "none";
    document.getElementById("prediction-loading").style.display = "block";
    fetch("/api/prediction/" + encodeURIComponent(area))
        .then(function (r) { return r.json(); })
        .then(function (resp) {
            document.getElementById("prediction-loading").style.display = "none";
            if (!resp.success) {
                showPredictionError(resp.error || "Prediction failed");
                return;
            }
            renderPrediction(resp.data);
        })
        .catch(function (err) {
            document.getElementById("prediction-loading").style.display = "none";
            showPredictionError("Error: " + err.message);
        });
}

function showPredictionError(msg) {
    var el = document.getElementById("prediction-error");
    el.textContent = msg;
    el.style.display = "block";
}

function renderPrediction(data) {
    var summary = data.summary;
    var predictions = data.predictions;
    document.getElementById("selected-area").textContent = data.area;
    setVal("p-min-temp", summary.min_temperature);
    setVal("p-max-temp", summary.max_temperature);
    setVal("p-avg-temp", summary.avg_temperature);
    setVal("p-humidity", summary.avg_humidity);
    setVal("p-rainfall", summary.total_rainfall);
    setVal("p-wind", summary.avg_wind_speed);
    setVal("p-rain-prob", summary.avg_rain_probability);
    renderPredictionTable(predictions);
    renderPredictionCharts(predictions);
    document.getElementById("prediction-content").style.display = "block";
}

function setVal(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val != null ? val : "--";
}

function renderPredictionTable(predictions) {
    var tbody = document.getElementById("prediction-tbody");
    tbody.innerHTML = "";
    predictions.forEach(function (p) {
        var tr = document.createElement("tr");
        tr.innerHTML =
            '<td>' + p.date + '</td>' +
            '<td>' + p.temperature + '</td>' +
            '<td>' + p.min_temperature + '</td>' +
            '<td>' + p.max_temperature + '</td>' +
            '<td>' + p.humidity + '</td>' +
            '<td>' + p.wind_speed + '</td>' +
            '<td>' + p.rainfall + '</td>' +
            '<td>' + p.weather_condition + '</td>' +
            '<td>' + p.rain_probability + '%</td>';
        tbody.appendChild(tr);
    });
}

function renderPredictionCharts(predictions) {
    renderRainProbChart(predictions);
    renderTempForecastChart(predictions);
}

/* ---- Theme-aware chart colors ---- */
function getThemeColors() {
    var isDark = document.documentElement.getAttribute("data-theme") === "dark";
    return {
        primary: "#4F46E5",
        primaryLight: "rgba(79,70,229,0.7)",
        primaryMuted: "rgba(79,70,229,0.3)",
        teal: "#14B8A6",
        tealSoft: "rgba(20,184,166,0.1)",
        red: "#EF4444",
        redSoft: "rgba(239,68,68,0.1)",
        gridLine: isDark ? "rgba(51,65,85,0.6)" : "rgba(226,232,240,0.6)",
        tooltipBg: "#1E293B",
        labelColor: isDark ? "#94A3B8" : "#64748B"
    };
}

var chartFont = { family: "'Inter', sans-serif", size: 12, weight: "500" };
var chartColors = getThemeColors();

Chart.defaults.font = chartFont;
Chart.defaults.color = chartColors.labelColor;

var rainProbChart = null;
var tempForecastChart = null;

function renderRainProbChart(predictions) {
    var ctx = document.getElementById("chart-rain-prob");
    if (!ctx) return;
    if (rainProbChart) rainProbChart.destroy();
    var labels = predictions.map(function (p) {
        var d = new Date(p.date);
        return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
    });
    var c = chartColors;
    rainProbChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Rain Probability (%)",
                data: predictions.map(function (p) { return p.rain_probability; }),
                backgroundColor: predictions.map(function (p) {
                    return p.rain_probability > 50 ? c.primaryLight : c.primaryMuted;
                }),
                borderColor: c.primary,
                borderWidth: 1.5, borderRadius: 6, borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: {
                legend: { position: "top", labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } },
                tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8, callbacks: { label: function (c) { return "Probability: " + c.raw + "%"; } } }
            },
            scales: { y: { min: 0, max: 100, grid: { color: c.gridLine }, title: { display: true, text: "Probability (%)", font: { weight: "600" } } }, x: { grid: { display: false } } }
        }
    });
}

function renderTempForecastChart(predictions) {
    var ctx = document.getElementById("chart-temp-forecast");
    if (!ctx) return;
    if (tempForecastChart) tempForecastChart.destroy();
    var labels = predictions.map(function (p) {
        var d = new Date(p.date);
        return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
    });
    var c = chartColors;
    tempForecastChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                { label: "Max Temperature", data: predictions.map(function (p) { return p.max_temperature; }), borderColor: c.red, backgroundColor: c.redSoft, fill: "+1", tension: 0.35, borderWidth: 2, pointRadius: 3, pointBackgroundColor: c.red },
                { label: "Avg Temperature", data: predictions.map(function (p) { return p.temperature; }), borderColor: c.primary, fill: false, tension: 0.35, borderWidth: 2.5, pointRadius: 3, pointBackgroundColor: c.primary },
                { label: "Min Temperature", data: predictions.map(function (p) { return p.min_temperature; }), borderColor: c.teal, backgroundColor: c.tealSoft, fill: "-1", tension: 0.35, borderWidth: 2, pointRadius: 3, pointBackgroundColor: c.teal }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "circle", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { y: { grid: { color: c.gridLine }, title: { display: true, text: "Temperature (C)", font: { weight: "600" } } }, x: { grid: { display: false } } }
        }
    });
}
