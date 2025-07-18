document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("simulateForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    // 성별 버튼 클릭
    document.querySelectorAll(".gender-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const value = btn.getAttribute("data-value");
            document.getElementById("gender").value = value;

            // 버튼 스타일 활성화
            document.querySelectorAll(".gender-btn").forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
        });
    });

    form.addEventListener("submit", (e) => {
        // 기본 제출 막음
        e.preventDefault();

        const birthInput = document.getElementById("birth");
        const birth = parseInt(birthInput.value);
        const nowYear = new Date().getFullYear();
        const age = nowYear - birth;
        document.getElementById("age").value = age;

        // 필수 필드 유효성 검사
        const requiredFields = document.querySelectorAll('#simulateForm [required]');
        for (let i = 0; i < requiredFields.length; i++) {
            const field = requiredFields[i];
            if (field.type === 'radio') {
                const radioGroup = document.getElementsByName(field.name);
                let isChecked = false;
                for (let j = 0; j < radioGroup.length; j++) {
                    if (radioGroup[j].checked) {
                        isChecked = true;
                        break;
                    }
                }
                if (!isChecked) {
                    alert('모든 필수 항목을 입력해주세요.');
                    return;
                }
            } else if (field.value.trim() === '') {
                alert('모든 필수 항목을 입력해주세요.');
                field.focus();
                return;
            }
        }

        // 성별 hidden 필드 선택 여부 확인
        if (document.getElementById("gender").value === "") {
            alert('성별을 선택해주세요.');
            return;
        }

        // 연령 유효성 검사 (적절한 범위 설정)
        if (age < 15 || age > 90) { // 예시: 15세 미만, 90세 이상은 비정상으로 간주
            alert('유효한 출생년도를 입력해주세요. (예: 1995 -> 나이 범위 15~90세)');
            birthInput.focus();
            return;
        }

        // 로딩 오버레이 표시
        loadingOverlay.style.display = "flex";
        
        // 폼 제출
        form.submit();
    });
});