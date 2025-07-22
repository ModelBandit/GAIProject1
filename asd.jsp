<%@ page contentType="text/html; charset=UTF-8" language="java"%>
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>사업차 상업부문 지역 분석</title>
<link rel="stylesheet" href="css/main.css">
<script
	src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<style>
.analysis-container {
	max-width: 1200px;
	margin: auto;
	padding: 40px 20px;
}

.page-title {
	text-align: center;
	font-size: 2.5rem;
	margin-bottom: 40px;
}

.main-content {
	display: flex;
	gap: 30px;
}

.sidebar {
	width: 300px;
	background: #fff;
	border-radius: 15px;
	padding: 20px;
	box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.chart-area {
	flex-grow: 1;
}

.industry-item {
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 8px 0;
	border-bottom: 1px solid #ddd;
}

.industry-color {
	width: 16px;
	height: 16px;
	border-radius: 50%;
	margin-left: 8px;
}

.remove-btn {
	background: #e74c3c;
	color: #fff;
	border: none;
	border-radius: 50%;
	width: 20px;
	height: 20px;
	cursor: pointer;
	font-size: 14px;
	line-height: 1;
}

.add-btn {
	margin-top: 10px;
	padding: 8px;
	width: 100%;
}

.industry-list-wrapper {
	max-height: 200px;
	overflow-y: auto;
	margin-top: 15px;
	border: 1px solid #eee;
	border-radius: 8px;
	padding: 10px;
}
</style>
</head>
<body>
	<%@ include file="common/header.jsp"%>
	<div class="analysis-container">
		<h1 class="page-title">사업차 상업부문 지역 분석</h1>
		<div class="main-content">
			<!-- Sidebar -->
			<div class="sidebar">
				<h3>선택된 상업</h3>
				<select id="industrySelect" style="width: 100%; padding: 8px;">
					<option value="">-- 상업 선택 --</option>
					<option value="agriculture">농에업</option>
					<option value="manufacturing">제조업</option>
					<option value="it">정보화</option>
				</select>
				<button id="addIndustryBtn" class="add-btn">+추가</button>
				<div class="industry-list-wrapper" id="industryList"></div>
			</div>

			<!-- Chart -->
			<div class="chart-area">
				<canvas id="industryChart"></canvas>
			</div>
		</div>
	</div>
	<script>
const ctx = document.getElementById('industryChart').getContext('2d');
const industryData = {
    total: { name: '전체상업', data: [2.8,3.2,3.5,3.1,2.9], color: '#3498db' },
    agriculture: { name: '농에업', data: [1.2,1.4,1.5,1.3,1.1], color: '#2ecc71' },
    manufacturing: { name: '제조업', data: [3.0,3.3,3.6,3.2,3.1], color: '#e74c3c' },
    it: { name: '정보화', data: [4.1,4.5,4.9,4.6,4.3], color: '#9b59b6' }
};

let selectedIndustries = ['total'];
let industryChart = new Chart(ctx, {
    type: 'line',
    data: { labels: ['2025','2026','2027','2028','2029'], datasets: [] },
    options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: {
            y: {
                ticks: { callback: val => val + '%' },
                min: 0, max: 6
            }
        }
    }
});

function updateChart() {
    industryChart.data.datasets = selectedIndustries.map(key => {
        const item = industryData[key];
        return {
            label: item.name,
            data: item.data,
            borderColor: item.color,
            backgroundColor: item.color + '20',
            fill: true,
            tension: 0.3
        };
    });
    industryChart.update();
}

function updateList() {
    const list = document.getElementById('industryList');
    list.innerHTML = '';
    selectedIndustries.forEach(key => {
        if (key === 'total') return;
        const item = document.createElement('div');
        item.className = 'industry-item';
        item.innerHTML = `
            <span>${industryData[key].name}</span>
            <div>
                <span class="industry-color" style="background:${industryData[key].color}"></span>
                <button class="remove-btn" onclick="removeIndustry('${key}')">&times;</button>
            </div>
        `;
        list.appendChild(item);
    });
}

function removeIndustry(key) {
    selectedIndustries = selectedIndustries.filter(k => k !== key);
    updateChart();
    updateList();
}

document.getElementById('addIndustryBtn').addEventListener('click', () => {
    const value = document.getElementById('industrySelect').value;
    if (value && !selectedIndustries.includes(value)) {
        selectedIndustries.push(value);
        updateChart();
        updateList();
    }
});

// Init
updateChart();
updateList();
</script>
	<%@ include file="common/footer.jsp"%>
</body>
</html>