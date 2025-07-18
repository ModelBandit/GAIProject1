<%@ page contentType="text/html; charset=UTF-8"%>
<%
  String user = (String) session.getAttribute("user");
%>
<link rel="stylesheet" href="css/main.css">
<link rel="stylesheet" href="css/simulate.css">

<header>
  <div class="logo">
    <a href="main.jsp" class="logo">CAREER<span>.AI</span></a>
  </div>

  <button class="hamburger" id="hamburger">&#9776;</button>

  <nav>
    <ul class="nav-menu" id="navMenu">
      <li><a href="simulate.jsp">Simulate</a></li>
      <li><a href="advice.jsp">AI Advice</a></li>
      <li><a href="about.jsp">About</a></li>
      <li><a href="faq.jsp">FAQ</a></li>

      <% if (user == null) { %>
        <li><a href="login.jsp" class="btn-link">로그인</a></li>
        <li><a href="signup.jsp" class="btn-link">회원가입</a></li>
      <% } else { %>
        <li><span class="btn-link" style="cursor:default;"><%= user %>님</span></li>
        <li><a href="logout.jsp" class="btn-link">로그아웃</a></li>
      <% } %>
    </ul>
  </nav>
</header>
