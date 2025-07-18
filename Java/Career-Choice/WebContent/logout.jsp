<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%
  session.invalidate(); // 세션 초기화 (로그아웃 처리)
  response.sendRedirect("main.jsp?logout=1"); // 로그아웃 후 메인으로 이동
%>