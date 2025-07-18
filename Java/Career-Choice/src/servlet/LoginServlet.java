package servlet;

import java.io.IOException;
import java.sql.*;
import javax.servlet.*;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

@WebServlet("/login")
public class LoginServlet extends HttpServlet {

  private static final String DB_URL = "jdbc:oracle:thin:@localhost:1521:xe";
  private static final String DB_USER = "MIN";
  private static final String DB_PASSWORD = "min";

  @Override
  protected void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    request.setCharacterEncoding("UTF-8");

    String username = request.getParameter("username");
    String password = request.getParameter("password");

    boolean loginSuccess = false;

    try (
      Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
      PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM members WHERE username = ? AND password = ?")
    ) {
      Class.forName("oracle.jdbc.driver.OracleDriver");

      pstmt.setString(1, username);
      pstmt.setString(2, password);

      ResultSet rs = pstmt.executeQuery();

      if (rs.next()) {
        loginSuccess = true;
        // 세션에 사용자 정보 저장
        HttpSession session = request.getSession();
        session.setAttribute("user", username);
      }

    } catch (Exception e) {
      e.printStackTrace();
    }

    System.out.println("[▶] 로그인 시도: " + username);

    if (loginSuccess) {
      System.out.println("[✔] 로그인 성공 → main.jsp 리디렉션");
      response.sendRedirect("main.jsp");
    } else {
      System.out.println("[✘] 로그인 실패");
      request.setAttribute("error", "아이디 또는 비밀번호가 올바르지 않습니다.");
      RequestDispatcher rd = request.getRequestDispatcher("login.jsp");
      rd.forward(request, response);
    }
  }
}
