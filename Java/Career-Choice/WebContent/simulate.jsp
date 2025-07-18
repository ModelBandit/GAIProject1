<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<%@ include file="common/header.jsp"%>
<html>
<head>
    <title>커리어 선택 시뮬레이션 | CAREER.AI</title>
    <link rel="stylesheet" href="css/simulate.css">
    <script src="js/main.js" defer></script>
</head>
<body>
    <div class="simulate-wrapper">
        <h2>커리어 선택 시뮬레이션</h2>
        <%-- 오류 메시지 표시 --%>
        <% String errorMessage = (String) request.getAttribute("errorMessage"); %>
        <% if (errorMessage != null && !errorMessage.isEmpty()) { %>
            <p style="color: red; font-weight: bold;"><%= errorMessage %></p>
        <% } %>

        <form id="simulateForm" action="${pageContext.request.contextPath}/SimulateServlet" method="post" class="simulate-form">
            <div class="form-group">
                <label for="birth">출생년도 (4자리)</label>
                <input type="number" id="birth" name="birth" placeholder="예: 1995" required />
                <input type="hidden" name="age" id="age" />
            </div>

            <div class="form-group">
                <label>성별</label>
                <div class="gender-buttons">
                    <button type="button" class="gender-btn" data-value="0">남성</button>
                    <button type="button" class="gender-btn" data-value="1">여성</button>
                </div>
                <input type="hidden" name="gender" id="gender" required>
            </div>

            <div class="form-group">
                <label for="education">최종 학력</label>
                <select id="education" name="education" required>
                    <option value="">선택</option>
                    <option value="1">무학 또는 초중퇴</option>
                    <option value="2">고졸 미만</option>
                    <option value="3">고졸</option>
                    <option value="4">대재 또는 중퇴</option>
                    <option value="5">전문대 졸업</option>
                    <option value="6">대학교 졸업 이상</option>
                </select>
            </div>

            <div class="form-group">
                <label for="current_job_category">현재 직업군</label>
                <select id="current_job_category" name="current_job_category" required>
                    <option value="">선택</option>
                    <option value="1">관리자</option>
                    <option value="2">전문가 및 관련 종사자</option>
                    <option value="3">사무 종사자</option>
                    <option value="4">서비스 종사자</option>
                    <option value="5">판매 종사자</option>
                    <option value="6">농림어업 숙련 종사자</option>
                    <option value="7">기능원 및 관련 기능 종사자</option>
                    <option value="8">장치·기계조작 및 조립 종사자</option>
                    <option value="9">단순노무 종사자</option>
                </select>
            </div>

            <div class="form-group">
                <label for="monthly_income">현재 월 소득 (만원)</label>
                <input type="number" id="monthly_income" name="monthly_income" placeholder="예: 300" inputmode="numeric" required />
            </div>

            <div class="form-group">
                <label>현재 직무 만족도 (1~5점)</label>
                <div class="radio-group">
                    <label><input type="radio" name="job_satisfaction" value="1" required> 1</label>
                    <label><input type="radio" name="job_satisfaction" value="2"> 2</label>
                    <label><input type="radio" name="job_satisfaction" value="3"> 3</label>
                    <label><input type="radio" name="job_satisfaction" value="4"> 4</label>
                    <label><input type="radio" name="job_satisfaction" value="5"> 5</label>
                </div>
            </div>

            <div class="form-group">
                <label for="satis_focus_key">가장 중요하게 생각하는 만족도 요인</label>
                <select id="satis_focus_key" name="satis_focus_key" required>
                    <option value="">선택</option>
                    <option value="satis_wage">임금 또는 보수</option>
                    <option value="satis_stability">고용 안정성</option>
                    <option value="satis_task_content">일의 내용</option>
                    <option value="satis_work_env">근무 환경</option>
                    <option value="satis_work_time">근로 시간</option>
                    <option value="satis_growth">성장 가능성</option>
                    <option value="satis_communication">인간관계 및 소통</option>
                    <option value="satis_fair_eval">인사고과의 공정성</option>
                    <option value="satis_welfare">복지후생 제도</option>
                </select>
            </div>

            <button type="submit" class="submit-btn">예측하기</button>
        </form>
    </div>

    <div id="loadingOverlay" style="display:none;">
        <div class="spinner"></div>
        <p>예측 중입니다. 잠시만 기다려주세요</p>
    </div>

    <script src="js/simulate.js"></script>
    <%@ include file="common/footer.jsp"%>
</body>
</html>
