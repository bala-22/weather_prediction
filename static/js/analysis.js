/**
 * Analysis page JavaScript
 * Loads historical data, renders charts, metrics switcher, algorithm comparison,
 * and feature importances for the 6-Month Analysis page.
 * Theme-aware: reads CSS variables for dark/light mode chart colors.
 */

var fullMetrics = null;

document.addEventListener("DOMContentLoaded", function () {
    lucide.createIcons();
    loadAnalysisData();
});

function fmtDate(dateStr) {
    if (!dateStr) return dateStr;
    var parts = dateStr.split("-");
    if (parts.length === 3) return parts[2] + "/" + parts[1] + "/" + parts[0];
    if (parts.length === 2) {
        var monthNames = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        var mIdx = parseInt(parts[1], 10);
        return (monthNames[mIdx] || parts[1]) + " " + parts[0];
    }
    return dateStr;
}

function loadAnalysisData() {
    fetch("/api/historical-analysis")
        .then(function (r) { return r.json(); })
        .then(function (resp) {
            if (!resp.success) {
                showError(resp.error || "Failed to load analysis data");
                return;
            }
            var data = resp.data;
            fullMetrics = data.metrics;

            renderDashboardCards(fullMetrics);
            renderPeriodInfo(data.period);
            renderMonthlyTable(data.monthly);
            renderCharts(data);
            loadFailureAnalysis();
            document.getElementById("loading-indicator").style.display = "none";
            document.getElementById("analysis-content").style.display = "block";
            lucide.createIcons();
        })
        .catch(function (err) {
            showError("Error loading data: " + err.message);
        });
}

function loadFailureAnalysis() {
    fetch("/api/failure-analysis")
        .then(function (r) { return r.json(); })
        .then(function (resp) {
            if (resp.success && resp.data) {
                renderFailureTable(resp.data);
            }
        })
        .catch(function () {});
}

function renderFailureTable(failures) {
    if (!failures || !failures.length) return;
    var fpCount = failures.filter(function (f) { return f.error_type.indexOf("FP") >= 0; }).length;
    var fnCount = failures.filter(function (f) { return f.error_type.indexOf("FN") >= 0; }).length;
    var summary = document.getElementById("failure-summary");
    if (summary) {
        summary.innerHTML =
            '<div class="failure-stat"><span class="failure-count fp-count">' + fpCount + '</span><span class="failure-label">False Positives (predicted rain, but stayed dry)</span></div>' +
            '<div class="failure-stat"><span class="failure-count fn-count">' + fnCount + '</span><span class="failure-label">False Negatives (missed sudden convective rain)</span></div>' +
            '<div class="failure-stat"><span class="failure-count total-count">' + failures.length + '</span><span class="failure-label">Total Divergent Days (out of 182)</span></div>';
    }
    var tbody = document.getElementById("failure-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    failures.forEach(function (f) {
        var tr = document.createElement("tr");
        var typeClass = f.error_type.indexOf("FP") >= 0 ? "fp-row" : "fn-row";
        tr.className = typeClass;
        var reason = f.reason || "";
        var reasonHtml = reason.replace(/\n/g, "<br>");
        tr.innerHTML =
            '<td>' + fmtDate(f.date) + '</td>' +
            '<td><span class="error-badge ' + typeClass + '">' + f.error_type + '</span></td>' +
            '<td>' + f.temperature + '</td>' +
            '<td>' + f.humidity + '%</td>' +
            '<td>' + f.rainfall + ' mm</td>' +
            '<td>' + f.cloud_cover + '%</td>' +
            '<td>' + f.weather_condition + '</td>' +
            '<td class="reason-cell">' + reasonHtml + '</td>';
        tbody.appendChild(tr);
    });
}

function showError(msg) {
    var loading = document.getElementById("loading-indicator");
    if (loading) loading.style.display = "none";
    var el = document.getElementById("error-message");
    if (el) {
        el.textContent = msg;
        el.style.display = "block";
    }
}

function renderPeriodInfo(period) {
    if (!period) return;
    var startEl = document.getElementById("period-start");
    var endEl = document.getElementById("period-end");
    var totalEl = document.getElementById("period-total");
    if (startEl) startEl.textContent = fmtDate(period.start);
    if (endEl) endEl.textContent = fmtDate(period.end);
    if (totalEl) totalEl.textContent = period.total_days;
}

function renderDashboardCards(m) {
    if (!m) return;
    setCard("a-total", m.total_days);
    setCard("a-predicted", m.predicted_rain_days);
    setCard("a-actual", m.actual_rain_days);
    var cm = m.confusion_matrix || {};
    var tpVal = cm.tp != null ? cm.tp : (m.true_positives != null ? m.true_positives : "--");
    var tnVal = cm.tn != null ? cm.tn : (m.true_negatives != null ? m.true_negatives : "--");
    var fpVal = cm.fp != null ? cm.fp : (m.false_positives != null ? m.false_positives : "--");
    var fnVal = cm.fn != null ? cm.fn : (m.false_negatives != null ? m.false_negatives : "--");

    setCard("a-rain-pct", m.rain_prediction_percentage != null ? m.rain_prediction_percentage + "%" : "--");
    setCard("a-actual-pct", m.actual_rain_percentage != null ? m.actual_rain_percentage + "%" : "--");

    renderCmCell("cm-tp", tpVal, m.total_days);
    renderCmCell("cm-tn", tnVal, m.total_days);
    renderCmCell("cm-fp", fpVal, m.total_days);
    renderCmCell("cm-fn", fnVal, m.total_days);
}

function renderCmCell(id, value, total) {
    var el = document.getElementById(id);
    if (!el) return;
    var valEl = el.querySelector(".cm-value");
    var pctEl = el.querySelector(".cm-pct");
    if (valEl) valEl.textContent = value != null ? value : "--";
    if (pctEl && total && typeof value === "number") {
        pctEl.textContent = ((value / total) * 100).toFixed(1) + "% of days";
    }
}

function setCard(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val != null ? val : "--";
}

function renderMonthlyTable(monthly) {
    if (!monthly || !monthly.length) return;
    var tbody = document.getElementById("monthly-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    monthly.forEach(function (row) {
        var tr = document.createElement("tr");
        var acc = row.accuracy != null ? row.accuracy : "--";
        var accStyle = row.accuracy >= 90 ? "color:var(--success-dark);font-weight:700;" : "color:var(--primary);font-weight:700;";
        tr.innerHTML =
            "<td><strong>" + fmtDate(row.month) + "</strong></td>" +
            "<td><span class=\"badge-climate\">" + (row.season || "Season") + "</span><br><small style=\"color:var(--text-muted);font-size:0.75rem;\">" + (row.description || "") + "</small></td>" +
            "<td>" + row.total_days + "</td>" +
            "<td><span style=\"font-weight:700;\">" + row.rain_days + "</span></td>" +
            "<td><span style=\"font-weight:700; color:var(--teal);\">" + (row.predicted_rain_days != null ? row.predicted_rain_days : "--") + "</span></td>" +
            "<td><span style=\"" + accStyle + "\">" + acc + "%</span></td>" +
            "<td>" + row.total_rainfall + " mm</td>" +
            "<td>" + row.avg_temperature + " &deg;C</td>" +
            "<td>" + row.avg_humidity + "%</td>";
        tbody.appendChild(tr);
    });
}

function renderCharts(data) {
    renderRainComparisonChart(data);
    renderMonthlyRainfallChart(data.monthly);
    renderMonthlyTempChart(data.monthly);
    renderTemperatureChart(data.daily);
    renderHumidityChart(data.daily);
    renderRainfallTrendChart(data.daily);
}

/* ---- Theme-aware chart colors ---- */
function getThemeColors() {
    var isDark = document.documentElement.getAttribute("data-theme") === "dark";
    return {
        primary: "#4F46E5",
        primaryLight: "rgba(79,70,229,0.7)",
        primarySoft: "rgba(79,70,229,0.1)",
        teal: "#14B8A6",
        tealLight: "rgba(20,184,166,0.7)",
        tealSoft: "rgba(20,184,166,0.1)",
        indigo: "#6366F1",
        indigoLight: "rgba(99,102,241,0.7)",
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

function renderRainComparisonChart(data) {
    var ctx = document.getElementById("chart-rain-comparison");
    if (!ctx) return;
    var daily = data.daily || [];
    var monthly = {};
    daily.forEach(function (d) {
        var m = d.date.substring(0, 7);
        if (!monthly[m]) monthly[m] = { actual: 0, predicted: 0 };
        if (d.is_rain) monthly[m].actual++;
        if (d.has_prediction !== false && d.predicted_rain) monthly[m].predicted++;
    });
    var months = Object.keys(monthly).sort();
    var monthNames = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var labels = months.map(function (m) {
        var parts = m.split("-");
        var mIdx = parseInt(parts[1], 10);
        return (monthNames[mIdx] || parts[1]) + " " + parts[0];
    });
    var actualCounts = months.map(function (m) { return monthly[m].actual; });
    var predictedCounts = months.map(function (m) { return monthly[m].predicted; });
    var c = chartColors;
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                { label: "Actual Rain Days", data: actualCounts, backgroundColor: c.primaryLight, borderColor: c.primary, borderWidth: 1.5, borderRadius: 6, borderSkipped: false },
                { label: "Predicted Rain Days", data: predictedCounts, backgroundColor: c.tealLight, borderColor: c.teal, borderWidth: 1.5, borderRadius: 6, borderSkipped: false }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: {
                legend: { position: "top", labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } },
                tooltip: { backgroundColor: c.tooltipBg, titleFont: { weight: "600" }, padding: 12, cornerRadius: 8, displayColors: true,
                    callbacks: { label: function (ctx) { return ctx.dataset.label + ": " + ctx.raw + " days"; } }
                }
            },
            scales: {
                y: { beginAtZero: true, min: 0, ticks: { stepSize: 2 }, grid: { color: c.gridLine }, title: { display: true, text: "Rain Days", font: { weight: "600" } } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderMonthlyRainfallChart(monthly) {
    var ctx = document.getElementById("chart-monthly-rainfall");
    if (!ctx || !monthly || !monthly.length) return;
    var c = chartColors;
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: monthly.map(function (m) { return fmtDate(m.month); }),
            datasets: [{ label: "Total Rainfall (mm)", data: monthly.map(function (m) { return m.total_rainfall; }), backgroundColor: c.indigoLight, borderColor: c.indigo, borderWidth: 1.5, borderRadius: 6, borderSkipped: false }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { y: { beginAtZero: true, grid: { color: c.gridLine }, title: { display: true, text: "Rainfall (mm)", font: { weight: "600" } } }, x: { grid: { display: false } } }
        }
    });
}

function renderMonthlyTempChart(monthly) {
    var ctx = document.getElementById("chart-monthly-temp");
    if (!ctx || !monthly || !monthly.length) return;
    var c = chartColors;
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: monthly.map(function (m) { return fmtDate(m.month); }),
            datasets: [{
                label: "Avg Temperature (°C)",
                data: monthly.map(function (m) { return m.avg_temperature; }),
                backgroundColor: c.indigoLight,
                borderColor: c.indigo, borderWidth: 1.5, borderRadius: 6, borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { y: { beginAtZero: true, grid: { color: c.gridLine }, title: { display: true, text: "Temperature (°C)", font: { weight: "600" } } }, x: { grid: { display: false } } }
        }
    });
}

function renderTemperatureChart(daily) {
    var ctx = document.getElementById("chart-temperature");
    if (!ctx || !daily || !daily.length) return;
    var labels = daily.map(function (d) { return fmtDate(d.date); });
    var c = chartColors;
    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                { label: "Avg Temperature", data: daily.map(function (d) { return d.temperature; }), borderColor: c.primary, backgroundColor: c.primarySoft, fill: false, tension: 0.35, pointRadius: 1, borderWidth: 2.5 },
                { label: "Min Temperature", data: daily.map(function (d) { return d.min_temperature; }), borderColor: c.teal, fill: false, tension: 0.35, pointRadius: 0, borderDash: [4, 4], borderWidth: 1.5 },
                { label: "Max Temperature", data: daily.map(function (d) { return d.max_temperature; }), borderColor: c.red, fill: false, tension: 0.35, pointRadius: 0, borderDash: [4, 4], borderWidth: 1.5 }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "circle", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { x: { display: false }, y: { grid: { color: c.gridLine }, title: { display: true, text: "Temperature (°C)", font: { weight: "600" } } } }
        }
    });
}

function renderHumidityChart(daily) {
    var ctx = document.getElementById("chart-humidity");
    if (!ctx || !daily || !daily.length) return;
    var c = chartColors;
    new Chart(ctx, {
        type: "line",
        data: {
            labels: daily.map(function (d) { return fmtDate(d.date); }),
            datasets: [{
                label: "Humidity (%)",
                data: daily.map(function (d) { return d.humidity; }),
                borderColor: c.teal,
                backgroundColor: c.tealSoft,
                fill: true, tension: 0.35, pointRadius: 1, borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "circle", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { x: { display: false }, y: { min: 0, max: 100, grid: { color: c.gridLine }, title: { display: true, text: "Humidity (%)", font: { weight: "600" } } } }
        }
    });
}

function renderRainfallTrendChart(daily) {
    var ctx = document.getElementById("chart-rainfall-trend");
    if (!ctx || !daily || !daily.length) return;
    var c = chartColors;
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: daily.map(function (d) { return fmtDate(d.date); }),
            datasets: [{
                label: "Rainfall (mm)",
                data: daily.map(function (d) { return d.rainfall; }),
                backgroundColor: c.indigoLight,
                borderColor: c.indigo, borderWidth: 1, borderRadius: 4, borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: "easeOutQuart" },
            plugins: { legend: { position: "top", labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } }, tooltip: { backgroundColor: c.tooltipBg, padding: 12, cornerRadius: 8 } },
            scales: { x: { display: false }, y: { beginAtZero: true, grid: { color: c.gridLine }, title: { display: true, text: "Rainfall (mm)", font: { weight: "600" } } } }
        }
    });
}
