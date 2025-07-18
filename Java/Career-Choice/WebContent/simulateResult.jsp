<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ page import="java.util.*" %>
<%@ page import="org.json.JSONArray" %>
<%@ page import="org.json.JSONObject" %>
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" type="text/css" href="css/simulateResult.css">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>예측 결과 | CAREER.AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="js/main.js" defer></script>
</head>
<body>
    <%-- common/header.jsp 포함 --%>
    <%@ include file="../common/header.jsp" %>

    <div class="page-wrapper">

        <div class="sidebar">
            <h3>직업군 선택</h3>
            
            <%-- ================================================================== --%>
            <%--  JSP Scriptlet: 데이터 준비 (서블릿에서 전달된 데이터를 받음) --%>
            <%-- ================================================================== --%>
            <%
                // 직업군 코드와 이름 매핑
                Map<String, String> jobCategoryMap = new HashMap<>();
                jobCategoryMap.put("1", "관리자");
                jobCategoryMap.put("2", "전문가 및 관련 종사자");
                jobCategoryMap.put("3", "사무 종사자");
                jobCategoryMap.put("4", "서비스 종사자");
                jobCategoryMap.put("5", "판매 종사자");
                jobCategoryMap.put("6", "농림 어업 숙련 종사자");
                jobCategoryMap.put("7", "기능원 및 관련 기능 종사자");
                jobCategoryMap.put("8", "장치·기계조작 및 조립 종사자");
                jobCategoryMap.put("9", "단순 노무 종사자");

                // 만족도 요인 코드와 이름 매핑
                Map<String, String> satisFactorMap = new HashMap<>();
                satisFactorMap.put("satis_wage", "임금 또는 보수");
                satisFactorMap.put("satis_stability", "고용 안정성");
                satisFactorMap.put("satis_task_content", "일의 내용");
                satisFactorMap.put("satis_work_env", "근무 환경");
                satisFactorMap.put("satis_work_time", "근무 시간과 여가");
                satisFactorMap.put("satis_growth", "발전 가능성");
                satisFactorMap.put("satis_communication", "인간 관계");
                satisFactorMap.put("satis_fair_eval", "공정한 평가와 보상");
                satisFactorMap.put("satis_welfare", "복리 후생");
                
                // 사용자가 선택한 중요 만족도 요인
                String satisFocusKey = (String) request.getAttribute("satisFocusKey");
                String focusKeyName = satisFactorMap.get(satisFocusKey);
            %>
				
            <%-- ================================================================== --%>
            <%-- JSP Declaration: 헬퍼 함수 정의 (JSP 내에서 재사용될 함수) --%>
            <%-- ================================================================== --%>
			<%!
                // 소득 변화율 포맷팅 (e.g., 0.05 -> "+5.00%")
                String formatIncomeChange(double value) {
                    String formatted = String.format("%.2f", value * 100);
                    return (value > 0 ? "+" : "") + formatted + "%";
                }

                // 만족도 변화량 포맷팅 (e.g., 0.25 -> "+0.25점")
                String formatSatisfactionChange(double value) {
                    String formatted = String.format("%.2f", value);
                    return (value > 0 ? "+" : "") + formatted + "점";
                }

                // 값의 증감에 따라 다른 CSS 클래스 반환
                String getChangeClass(double value) {
                    if (value > 0) return "positive-change";
                    if (value < 0) return "negative-change";
                    return "no-change";
                }
        	%>

            <form id="simulationForm" action="SimulateServlet" method="post">
                <%-- hidden input들은 이전과 동일 --%>
                <input type="hidden" name="age" value="<%= request.getAttribute("userAge") %>">
                <input type="hidden" name="gender" value="<%= request.getAttribute("userGender") %>">
                <input type="hidden" name="education" value="<%= request.getAttribute("userEducation") %>">
                <input type="hidden" name="monthly_income" value="<%= request.getAttribute("userMonthlyIncome") %>">
                <input type="hidden" name="job_satisfaction" value="<%= request.getAttribute("userJobSatisfaction") %>">
                <input type="hidden" name="current_job_category" value="<%= request.getAttribute("currentJobCategory") %>">
                <input type="hidden" name="satis_focus_key" value="<%= request.getAttribute("satisFocusKey") %>">

                <div class="input-group">
                    <p><strong>현재 직업군:</strong> <%= jobCategoryMap.get(String.valueOf(request.getAttribute("currentJobCategory"))) %></p>

                    <label for="jobACategorySelect">선택 직업군 A</label>
                    <select id="jobACategorySelect" name="job_A_category" onchange="document.getElementById('simulationForm').submit()">
                        <%
                            for (Map.Entry<String, String> entry : jobCategoryMap.entrySet()) {
                                String categoryValue = entry.getKey();
                                String categoryName = entry.getValue();
                                String selected = String.valueOf(request.getAttribute("selectedJobACategory")).equals(categoryValue) ? "selected" : "";
                        %>
                        <option value="<%= categoryValue %>" <%= selected %>><%= categoryName %></option>
                        <%
                            }
                        %>
                    </select>
                </div>

                <div class="input-group">
                    <label for="jobBCategorySelect">선택 직업군 B</label>
                    <select id="jobBCategorySelect" name="job_B_category" onchange="document.getElementById('simulationForm').submit()">
                        <%
                            for (Map.Entry<String, String> entry : jobCategoryMap.entrySet()) {
                                String categoryValue = entry.getKey();
                                String categoryName = entry.getValue();
                                String selected = String.valueOf(request.getAttribute("selectedJobBCategory")).equals(categoryValue) ? "selected" : "";
                        %>
                        <option value="<%= categoryValue %>" <%= selected %>><%= categoryName %></option>
                        <%
                            }
                        %>
                    </select>
                </div>

                <!-- 추천 기준 조절 슬라이더 -->
                <div class="input-group">
                    <label for="prioritySlider">결과 추천 기준</label>
                    <div class="slider-wrapper">
                        <span>만족도 우선</span>
                        <input type="range" id="prioritySlider" min="0" max="100" value="50" step="10">
                        <span>소득 우선</span>
                    </div>
                    <div id="priorityLabel" class="slider-label">균형 (50:50)</div>
                </div>

                <div class="spinner" id="loadingSpinner" style="display: none;">
                    새로운 예측 결과를 불러오는 중입니다...
                </div>
            </form>
        </div>

        <div class="main-content">
            <h2>커리어 시뮬레이션 결과</h2>

            <%-- 오류 메시지 표시 --%>
            <% String errorMessage = (String) request.getAttribute("errorMessage"); %>
            <% if (errorMessage != null && !errorMessage.isEmpty()) { %>
                <p style="color:red;"><%= errorMessage %></p>
            <% } %>

            <h3>예측 결과 요약</h3>
            <div class="result-section" id="predictionResultsContainer">
                <%-- ================================================================== --%>
                <%-- JSP Scriptlet: 예측 결과 파싱 및 화면 표시 --%>
                <%-- ================================================================== --%>
                <%
                    JSONArray predictionResults = new JSONArray(String.valueOf(request.getAttribute("predictionResults")));
                    JSONObject currentResult = predictionResults.getJSONObject(0);
                    JSONObject jobAResult = predictionResults.getJSONObject(1);
                    JSONObject jobBResult = predictionResults.length() > 2 ? predictionResults.getJSONObject(2) : null;
                %>

                <div class="scenario-result">
                    <h4>현직 유지 시 예측</h4>
                    <p><strong>월 소득 변화율:</strong> <span class="highlight <%= getChangeClass(currentResult.getDouble("income_change_rate")) %>"><%= formatIncomeChange(currentResult.getDouble("income_change_rate")) %></span></p>
                    <p><strong>직무 만족도 변화:</strong> <span class="highlight <%= getChangeClass(currentResult.getDouble("satisfaction_change_score")) %>"><%= formatSatisfactionChange(currentResult.getDouble("satisfaction_change_score")) %></span>
                        <% if (focusKeyName != null) { %><span class="focus-key-info">(중요 항목: <%= focusKeyName %> 반영)</span><% } %>
                    </p>
                </div>

                <div class="scenario-result" id="jobAResultContainer">
                    <h4><span id="jobAName"><%= jobCategoryMap.get(String.valueOf(request.getAttribute("selectedJobACategory"))) %></span> 전직 시 예측</h4>
                    <p><strong>월 소득 변화율:</strong> <span class="highlight <%= getChangeClass(jobAResult.getDouble("income_change_rate")) %>" id="jobAIncome"><%= formatIncomeChange(jobAResult.getDouble("income_change_rate")) %></span></p>
                    <p><strong>직무 만족도 변화:</strong> <span class="highlight <%= getChangeClass(jobAResult.getDouble("satisfaction_change_score")) %>" id="jobASatis"><%= formatSatisfactionChange(jobAResult.getDouble("satisfaction_change_score")) %></span>
                        <% if (focusKeyName != null) { %><span class="focus-key-info">(중요 항목: <%= focusKeyName %> 반영)</span><% } %>
                    </p>
                </div>

                <div class="scenario-result" id="jobBResultContainer" style="<%= jobBResult == null ? "display:none;" : "" %>">
                    <h4><span id="jobBName"><%= jobBResult != null ? jobCategoryMap.get(String.valueOf(request.getAttribute("selectedJobBCategory"))) : "" %></span> 전직 시 예측</h4>
                    <p><strong>월 소득 변화율:</strong> <span class="highlight <%= getChangeClass(jobBResult != null ? jobBResult.getDouble("income_change_rate") : 0.0) %>" id="jobBIncome"><%= formatIncomeChange(jobBResult != null ? jobBResult.getDouble("income_change_rate") : 0.0) %></span></p>
                    <p><strong>직무 만족도 변화:</strong> <span class="highlight <%= getChangeClass(jobBResult != null ? jobBResult.getDouble("satisfaction_change_score") : 0.0) %>" id="jobBSatis"><%= formatSatisfactionChange(jobBResult != null ? jobBResult.getDouble("satisfaction_change_score") : 0.0) %></span>
                        <% if (jobBResult != null && focusKeyName != null) { %><span class="focus-key-info">(중요 항목: <%= focusKeyName %> 반영)</span><% } %>
                    </p>
                </div>
            </div>

            <h3>시나리오 비교</h3>
            <div class="charts-section">
                <div class="chart-container">
                    <h4>월 소득 변화율</h4>
                    <div class="chart-canvas-wrapper">
                        <canvas id="incomeChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h4>직무 만족도 변화</h4>
                    <div class="chart-canvas-wrapper">
                        <canvas id="satisfactionChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- AI 추천 섹션 -->
            <h3>AI 추천</h3>
            <div class="recommendation-section">
                <p><strong>추천 선택지:</strong> <span id="recommendedJobName" class="highlight">계산 중...</span></p>
                <p id="recommendationReason" class="recommendation-reason">추천 기준을 조절하여 가장 적합한 선택지를 찾아보세요.</p>
            </div>

            <form action="advice.jsp" method="get" style="text-align: center; margin-top: 20px;">
                <%-- AI 조언 페이지로 전달할 데이터 --%>
                <input type="hidden" name="userAge" value="<%= request.getAttribute("userAge") %>">
                <input type="hidden" name="userGender" value="<%= request.getAttribute("userGender") %>">
                <input type="hidden" name="userEducation" value="<%= request.getAttribute("userEducation") %>">
                <input type="hidden" name="userMonthlyIncome" value="<%= request.getAttribute("userMonthlyIncome") %>">
                <input type="hidden" name="userJobSatisfaction" value="<%= request.getAttribute("userJobSatisfaction") %>">
                <input type="hidden" name="currentJobCategoryName" value="<%= jobCategoryMap.get(String.valueOf(request.getAttribute("currentJobCategory"))) %>">
                <input type="hidden" name="selectedJobACategoryName" value="<%= jobCategoryMap.get(String.valueOf(request.getAttribute("selectedJobACategory"))) %>">
                <input type="hidden" name="selectedJobBCategoryName" value="<%= jobBResult != null ? jobCategoryMap.get(String.valueOf(request.getAttribute("selectedJobBCategory"))) : "" %>">
                <input type="hidden" name="satisFocusKeyName" value="<%= focusKeyName != null ? focusKeyName : "" %>">
                <input type="hidden" name="predictionResults" value='<%= request.getAttribute("predictionResults").toString().replace("'", "'") %>'>
                <input type="hidden" name="recommendedJobName" id="hiddenRecommendedJobName">
                <input type="hidden" name="recommendationReason" id="hiddenRecommendationReason">
                
                <button type="submit" class="btn-advice">AI 조언 받기</button>
            </form>

            <%-- ================================================================== --%>
            <%-- JSP Scriptlet: JS로 데이터 전달 (전역 변수 선언) --%>
            <%-- ================================================================== --%>
            <script>
                var predictionResultsRaw = JSON.parse('<%= request.getAttribute("predictionResults").toString().replace("\\", "\\\\").replace("'", "\\'") %>');
                var currentJobCategory = '<%= request.getAttribute("currentJobCategory") %>';
                var selectedJobACategory = '<%= request.getAttribute("selectedJobACategory") %>';
                var selectedJobBCategory = '<%= request.getAttribute("selectedJobBCategory") != null ? request.getAttribute("selectedJobBCategory").toString() : "" %>';

                <%
                    // jobCategoryMap을 JS 객체로 변환
                    StringBuilder jobMapJs = new StringBuilder("var jobCategoryMapJs = {");
                    for (Map.Entry<String, String> entry : jobCategoryMap.entrySet()) {
                        jobMapJs.append("\"").append(entry.getKey()).append("\": \"").append(entry.getValue()).append("\",");
                    }
                    if (jobCategoryMap.size() > 0) jobMapJs.setLength(jobMapJs.length() - 1);
                    jobMapJs.append("};");
                    out.println(jobMapJs.toString());
                %>
            </script>

            <script src="js/simulateResult.js"></script>
        </div>

    </div>
    
    <%-- common/footer.jsp 포함 --%>
    <%@ include file="../common/footer.jsp" %>
</body>
</html>