<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ page language="java" %>
<%@ include file="common/header.jsp"%>
<!DOCTYPE html>
<html>
<head>
<title>About | CAREER.AI</title>
<link rel="stylesheet" href="css/main.css">
<link rel="stylesheet" href="css/about.css">
<script src="js/main.js" defer></script>
</head>
<body>
	<div class="about-wrapper">
		<div class="section">
			<h1>CAREER.AI 서비스 소개</h1>
			<p>
				<strong>CAREER.AI</strong>는 커리어 전환 또는 현직 유지와 같은 중대한 선택을 앞둔 사용자가 <span
					class="highlight">예측 모델</span>과 <span class="highlight">생성형
					AI 조언</span>을 통해 보다 합리적이고 자기주도적인 결정을 내릴 수 있도록 지원하는 서비스입니다.
			</p>
		</div>

		<div class="section">
			<h2>주요 기능</h2>
			<ul>
				<li><strong>시뮬레이션:</strong> 사용자 조건 기반 전직/현직 유지 시의 예측 결과 제공</li>
				<li><strong>직군 트렌드 분석:</strong> 직업군별 성장률, 리스크 점수 등 노동시장 흐름 제공</li>
				<li><strong>생성형 AI 조언:</strong> GPT 기반 맞춤형 텍스트 피드백 제공</li>
				<li><strong>유사 사례 기반 예측:</strong> KLIPS 데이터를 기반으로 한 실제 사례 학습</li>
			</ul>
		</div>

		<div class="section">
			<h2>데이터 기반 및 예측 모델</h2>
			<p>
				CAREER.AI는 <strong>한국노동패널조사(KLIPS)</strong> 데이터를 기반으로, 사용자의 연령, 성별, 학력,
				직업군, 소득, 직무 만족도 정보를 활용해 유사 사례군을 구성하고, <strong>1년 후 소득 및 만족도
					변화</strong>를 예측합니다.
			</p>
			<p>결과는 소득 변화율(%) 및 직무 만족도 변화량(점수)으로 제공되며, 사용자가 중요하게 생각하는 기준(소득 또는
				만족도)에 따라 최적의 시나리오를 추천합니다.</p>
		</div>

		<div class="section">
			<h2>생성형 AI 조언</h2>
			<p>
				예측 결과를 기반으로, GPT 계열 언어모델이 <strong>‘경험 많은 선배’ 스타일의 조언</strong>을 자연어로
				제공합니다. 사용자의 커리어 목표에 맞춘 질문과 조언을 통해 자기 성찰과 결정을 돕습니다.
			</p>
		</div>

		<div class="section">
			<h2>주의사항 및 목표</h2>
			<ul>
				<li>CAREER.AI는 개인의 결정을 돕기 위한 <strong>보조 도구</strong>이며, 최종 결정은 사용자의
					몫입니다.
				</li>
				<li>예측 결과는 통계적 경향을 기반으로 하며, 모든 상황에 일치하는 것은 아닙니다.</li>
				<li>모든 사용자 정보는 익명화되며, <strong>안전하게 보호</strong>됩니다.
				</li>
			</ul>
		</div>
	</div>
	<%@ include file="common/footer.jsp"%>
</body>
</html>

