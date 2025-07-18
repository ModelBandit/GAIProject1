document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loginForm");

  form.addEventListener("submit", (e) => {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;

    if (!username || !password) {
      alert("아이디와 비밀번호를 모두 입력해주세요.");
      e.preventDefault();
      return;
    }

    // 서버 측 인증은 별도로 처리됨
  });
});
