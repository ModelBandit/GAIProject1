const ResultPageManager = {
    init: function () {
        this.cacheDOMElements();
        this.setupInitialData();
        this.createCharts();
        this.updateRecommendation();
        this.bindEventListeners();
    },

    cacheDOMElements: function () {
        this.elements = {
            prioritySlider: document.getElementById("prioritySlider"),
            priorityLabel: document.getElementById("priorityLabel"),
            recommendedJobName: document.getElementById("recommendedJobName"),
            recommendationReason: document.getElementById("recommendationReason"),
            incomeChartCanvas: document.getElementById('incomeChart'),
            satisfactionChartCanvas: document.getElementById('satisfactionChart'),
            adviceForm: document.querySelector('form[action="advice.jsp"]')
        };
    },

    setupInitialData: function () {
        this.scenarios = [
            {
                name: "현직 유지",
                income: predictionResultsRaw[0].income_change_rate,
                satisfaction: predictionResultsRaw[0].satisfaction_change_score,
            },
            {
                name: jobCategoryMapJs[selectedJobACategory] || "직업 A",
                income: predictionResultsRaw[1].income_change_rate,
                satisfaction: predictionResultsRaw[1].satisfaction_change_score,
            }
        ];

        if (predictionResultsRaw.length > 2 && predictionResultsRaw[2]) {
            this.scenarios.push({
                name: jobCategoryMapJs[selectedJobBCategory] || "직업 B",
                income: predictionResultsRaw[2].income_change_rate,
                satisfaction: predictionResultsRaw[2].satisfaction_change_score,
            });
        }

        this.generateUniqueScenarioNames();
    },

    generateUniqueScenarioNames: function () {
        const nameCount = {};
        this.scenarios = this.scenarios.map((s) => {
            const base = s.name;
            nameCount[base] = (nameCount[base] || 0) + 1;
            const newName = nameCount[base] > 1 ? `${base} #${nameCount[base]}` : base;
            return { ...s, name: newName };
        });
    },

    bindEventListeners: function () {
        this.elements.prioritySlider.addEventListener("input", () => this.updateRecommendation());
        if (this.elements.adviceForm) {
            this.elements.adviceForm.addEventListener("submit", (e) => {
                this.prepareAdviceData();
            });
        }
    },

    prepareAdviceData: function () {
        document.getElementById('hiddenRecommendedJobName').value = this.elements.recommendedJobName.textContent;
        document.getElementById('hiddenRecommendationReason').value = this.elements.recommendationReason.textContent;
    },

    updateRecommendation: function () {
        const incomeWeight = this.elements.prioritySlider.value / 100;
        const satisfactionWeight = 1 - incomeWeight;

        this.elements.priorityLabel.textContent = `소득 ${Math.round(incomeWeight * 100)}% : 만족도 ${Math.round(satisfactionWeight * 100)}%`;

        const minIncome = Math.min(...this.scenarios.map(s => s.income));
        const maxIncome = Math.max(...this.scenarios.map(s => s.income));
        const minSatisfaction = Math.min(...this.scenarios.map(s => s.satisfaction));
        const maxSatisfaction = Math.max(...this.scenarios.map(s => s.satisfaction));

        let bestScenario = null;
        let maxScore = -Infinity;

        this.scenarios.forEach(scenario => {
            const normalizedIncome = this.normalize(scenario.income, minIncome, maxIncome);
            const normalizedSatisfaction = this.normalize(scenario.satisfaction, minSatisfaction, maxSatisfaction);
            const score = (normalizedIncome * incomeWeight) + (normalizedSatisfaction * satisfactionWeight);

            if (score > maxScore) {
                maxScore = score;
                bestScenario = scenario;
            }
        });

        this.elements.recommendedJobName.textContent = bestScenario.name;
        this.updateRecommendationReason(incomeWeight);
    },

    updateRecommendationReason: function (incomeWeight) {
        let reason = "소득과 만족도의 균형을 고려했을 때 가장 안정적인 선택입니다.";
        if (incomeWeight > 0.7) {
            reason = "소득 상승을 가장 중요하게 고려했을 때 가장 유리한 선택입니다.";
        } else if (incomeWeight < 0.3) {
            reason = "직무 만족도 향상을 가장 중요하게 고려했을 때 가장 적합한 선택입니다.";
        }
        this.elements.recommendationReason.textContent = reason;
    },

    normalize: function (value, min, max) {
        if (max === min) return 0.5;
        return (value - min) / (max - min);
    },

    createCharts: function () {
        const labels = this.scenarios.map(s => s.name);
        const incomeData = this.scenarios.map(s => s.income * 100);
        const satisfactionData = this.scenarios.map(s => s.satisfaction);
        const count = this.scenarios.length;

        console.log("📊 시나리오 이름:", labels);
        console.log("💰 예측 소득 변화율 (%):", incomeData);
        console.log("🟣 예측 만족도 변화량:", satisfactionData);

        const bgColors = ['#7CB9E8', '#FFBC42', '#9ADE7B'].slice(0, count);
        const borderColors = ['#5C9BD8', '#DB9A00', '#70BA5F'].slice(0, count);

        const createChart = (canvas, chartLabel, data, bg, border, yAxisLabel) => {
            const dataMin = Math.min(...data);
            const dataMax = Math.max(...data);
            
            let yMin, yMax;

            // 데이터의 최소/최대값 범위를 기준으로 y축 범위를 설정하여 값의 차이를 명확하게 보여줍니다.
            const range = dataMax - dataMin;
            
            if (range === 0) {
                // 모든 데이터 값이 동일할 경우, 해당 값을 중심으로 위아래에 약간의 여백을 줍니다.
                const buffer = Math.abs(dataMax) * 0.2 || 1; // 20% 버퍼, 값이 0일 경우 1
                yMin = dataMin - buffer;
                yMax = dataMax + buffer;
            } else {
                // 데이터 값의 차이가 있을 경우, 전체 범위의 20%를 추가 여백으로 주어 가독성을 높입니다.
                const padding = range * 0.2;
                yMin = dataMin - padding;
                yMax = dataMax + padding;
            }

            // 이전 로직과 달리, 데이터가 모두 양수이거나 음수일 때 y축을 0에 고정하지 않습니다.
            // 이 변경으로 인해 값들이 서로 비슷하지만 0에서 멀리 떨어져 있을 때(예: 90, 95, 100)
            // 차트 상의 차이가 훨씬 명확하게 드러납니다.

            new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: labels,
                   

                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            type: 'category',  // ✅ 핵심: 인덱스 아닌 라벨로 강제
                            grid: { display: false },
                            ticks: {
                                autoSkip: false,
                                maxRotation: 45,
                                minRotation: 20,
                                callback: function (value, index, ticks) {
                                    // `this`는 scale 객체를 참조하므로 getLabelForValue를 사용해
                                    // 인덱스에 해당하는 실제 라벨을 가져옵니다.
                                    const label = this.getLabelForValue(value);
                                    return label.split(' ').join('\n');
                                }
                            }
                        },
                        y: {
                            min: yMin,
                            max: yMax,
                            ticks: {
                                callback: function (value) {
                                    return `${value.toFixed(2)}${yAxisLabel}`;
                                }
                            }
                        }
                    }
                }
            });
        };

        createChart(this.elements.incomeChartCanvas, '월 소득 변화율', incomeData, bgColors, borderColors, '%');
        createChart(this.elements.satisfactionChartCanvas, '직무 만족도 변화', satisfactionData, bgColors, borderColors, '점');
    }
};

document.addEventListener("DOMContentLoaded", () => ResultPageManager.init());