document.addEventListener("DOMContentLoaded", () => {
    const hamburger = document.getElementById("hamburger");
    const navMenu = document.getElementById("navMenu");

    hamburger.addEventListener("click", () => {
        navMenu.classList.toggle("active");
    });

    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('logout') && urlParams.get('logout') === '1') {
        alert("로그아웃 되었습니다.");
        // To avoid showing the alert on refresh, remove the query parameter from the URL
        if (window.history.replaceState) {
            const cleanUrl = window.location.protocol + "//" + window.location.host + window.location.pathname;
            window.history.replaceState({path: cleanUrl}, '', cleanUrl);
        }
    }
});