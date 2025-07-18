document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("signupForm");

  const idInput = document.getElementById("username");
  const emailInput = document.getElementById("email");
  const pwInput = document.getElementById("password");
  const confirmInput = document.getElementById("confirmPassword");

  const idMsg = document.getElementById("idCheckMsg");
  const emailMsg = document.getElementById("emailCheckMsg");
  const pwMsg = document.getElementById("pwMatchMsg");

  idInput.addEventListener("input", () => {
    const idPattern = /^[a-zA-Z0-9]{5,16}$/;
    idMsg.textContent = idPattern.test(idInput.value) ? "" : "5~16자 영문+숫자만 허용됩니다.";
  });

  emailInput.addEventListener("blur", () => {
    emailMsg.textContent = "";
  });

  confirmInput.addEventListener("input", () => {
    pwMsg.textContent = pwInput.value === confirmInput.value ? "" : "비밀번호가 일치하지 않습니다.";
  });

  form.addEventListener("submit", (e) => {
  if (pwInput.value !== confirmInput.value) {
    e.preventDefault();
    pwMsg.textContent = "비밀번호가 일치하지 않습니다.";
    return;
  }

  // 유효성 검사를 통과한 경우에는 e.preventDefault() 호출하지 않음!
});
});
