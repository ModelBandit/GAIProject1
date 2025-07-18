<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ page language="java"%>
<%@ include file="common/header.jsp"%>
<!DOCTYPE html>
<html>
<head>
<title>FAQ | CAREER.AI</title>
<link rel="stylesheet" href="css/main.css">
<link rel="stylesheet" href="css/faq.css">
<script src="js/main.js" defer></script>
</head>
<body>
	<div class="faq-wrapper">
		<h1>자주 묻는 질문 (FAQ)</h1>
		<div class="accordion">
			<div class="accordion-item">
				<button class="accordion-header">이 서비스는 무료인가요?</button>
				<div class="accordion-body">네, 현재 CAREER.AI는 모든 기능을 무료로 제공합니다. 일부
					고급 기능은 추후 유료화될 수 있으며, 사전 공지될 예정입니다.</div>
			</div>
			<div class="accordion-item">
				<button class="accordion-header">예측 결과는 얼마나 정확한가요?</button>
				<div class="accordion-body">KLIPS 데이터를 기반으로 학습된 모델이며, 평균적으로 높은
					설명력을 보입니다. 다만, 개별 상황에 따라 차이는 발생할 수 있습니다.</div>
			</div>
			<div class="accordion-item">
				<button class="accordion-header">제 정보를 저장하나요?</button>
				<div class="accordion-body">비회원 상태에서는 정보가 저장되지 않으며, 로그인 시에만
					선택적으로 저장됩니다. 저장된 정보는 모두 암호화되어 안전하게 보호됩니다.</div>
			</div>
			<div class="accordion-item">
				<button class="accordion-header">AI 조언은 어떻게 만들어지나요?</button>
				<div class="accordion-body">GPT 계열의 생성형 언어 모델이 예측 결과를 기반으로,
					맞춤형 텍스트 조언을 제공합니다.</div>
			</div>
			<div class="accordion-item">
				<button class="accordion-header">로그인을 꼭 해야 하나요?</button>
				<div class="accordion-body">아니요. 대부분 기능은 로그인 없이 사용 가능하며, 기록
					저장과 비교 기능은 로그인 시에 제공됩니다.</div>
			</div>
			<div class="accordion-item">
				<button class="accordion-header">결과가 마음에 들지 않아요. 다시 받을 수
					있나요?</button>
				<div class="accordion-body">예. 입력 조건을 바꾸거나 AI 조언을 재생성하여 언제든
					새로운 결과를 확인할 수 있습니다.</div>
			</div>
		</div>
	</div>

	<div class="email-section">
		<p>더 궁금한 점이 있으신가요?</p>
		<a href="mailto:contact@seedcareer.kr" class="email-btn">이메일 문의하기</a>
	</div>


	<script src="js/faq.js"></script>
	<%@ include file="common/footer.jsp"%>
</body>
</html>
