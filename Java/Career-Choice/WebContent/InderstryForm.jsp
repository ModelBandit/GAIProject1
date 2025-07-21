<%@ page contentType="text/html; charset=UTF-8" language="java"%>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>산업 분석 사용자 설문 | CAREER.AI</title>
  <link rel="stylesheet" href="css/main.css">
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #f9f9f9; margin: 0; padding: 0; }
    .container { max-width: 700px; margin: 40px auto; background: #fff; padding: 30px 40px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
    h1 { font-size: 1.8rem; margin-bottom: 20px; color: #2c3e50; text-align: center; }
    .form-group { margin-bottom: 25px; }
    label { font-weight: 600; display: block; margin-bottom: 10px; color: #34495e; }
    select, input[type=radio], input[type=range] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 1rem; }
    .radio-group, .range-labels { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
    .radio-group label { flex: 1; text-align: center; }
    .submit-btn { background: #3498db; color: white; font-weight: bold; padding: 12px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; width: 100%; }
    .submit-btn:hover { background: #2980b9; }
  </style>
</head>
<body>
  <div class="container">
    <h1>당신에게 맞는 산업을 찾기 위한 간단한 설문</h1>
    <form id="surveyForm">
      <div class="form-group">
        <label for="industry">관심 있는 산업을 선택하세요</label>
        <select id="industry" name="industry" required>
          <option value="">-- 산업 선택 --</option>
          <option value="agriculture">농업/임업/어업</option>
          <option value="mining">광업</option>
          <option value="manufacturing">제조업</option>
          <option value="utilities">전기/가스/수도</option>
          <option value="construction">건설업</option>
          <option value="wholesale">도매/소매업</option>
          <option value="transport">운수/창고업</option>
          <option value="hospitality">숙박/음식점업</option>
          <option value="it">정보통신업</option>
          <option value="finance">금융/보험업</option>
          <option value="realestate">부동산/임대업</option>
          <option value="professional">전문/과학/기술 서비스</option>
          <option value="education">교육 서비스업</option>
          <option value="healthcare">보건/사회복지</option>
          <option value="entertainment">예술/스포츠/여가</option>
          <option value="others">기타 개인서비스</option>
        </select>
      </div>

      <div class="form-group">
        <label>당신의 연령대는?</label>
        <div class="radio-group">
          <label><input type="radio" name="age" value="under30"> 20대 이하</label>
          <label><input type="radio" name="age" value="30s"> 30대</label>
          <label><input type="radio" name="age" value="40s"> 40대</label>
          <label><input type="radio" name="age" value="50plus"> 50대 이상</label>
        </div>
      </div>

      <div class="form-group">
        <label>보수와 고용 안정성 중 무엇이 더 중요하신가요?</label>
        <div class="radio-group">
          <label><input type="radio" name="priority" value="salary"> 보수</label>
          <label><input type="radio" name="priority" value="stability"> 고용 안정성</label>
          <label><input type="radio" name="priority" value="balance"> 균형</label>
        </div>
      </div>

      <div class="form-group">
        <label>근무 강도는 어느 정도를 선호하시나요?</label>
        <input type="range" id="workload" name="workload" min="1" max="5" value="3">
        <div class="range-labels">
          <span>짧게</span>
          <span>보통</span>
          <span>길게</span>
        </div>
      </div>

      <button type="submit" class="submit-btn">제출하고 분석 보기</button>
    </form>
  </div>

  <script>
    document.getElementById('surveyForm').addEventListener('submit', function(e) {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(this));
      console.log('사용자 입력:', data);
      // 이후 분석 페이지로 전달하거나, 분석 함수 호출 등 처리
      alert('입력이 완료되었습니다. 분석 결과로 이동합니다.');
      // window.location.href = '/result.html'; // 필요시 연결
    });
  </script>
</body>
</html>
