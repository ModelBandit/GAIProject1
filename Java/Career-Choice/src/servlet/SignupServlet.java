package servlet;

import java.io.IOException;
import java.sql.*;

import javax.servlet.*;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

@WebServlet("/signup")
public class SignupServlet extends HttpServlet {

	  private static final String DB_URL = "jdbc:oracle:thin:@localhost:1521:xe"; // 본인 환경에 맞게
	  private static final String DB_USER = "MIN";  // Oracle 계정명
	  private static final String DB_PASSWORD = "min"; // Oracle 비밀번호

  protected void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    request.setCharacterEncoding("UTF-8");

    String username = request.getParameter("username");
    String password = request.getParameter("password");
    String email = request.getParameter("email");

    boolean success = false;
    String errorMessage = null;

    try (
      Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
      PreparedStatement pstmt = conn.prepareStatement("INSERT INTO members (username, password, email) VALUES (?, ?, ?)")
    ) {
      Class.forName("oracle.jdbc.driver.OracleDriver");

      pstmt.setString(1, username);
      pstmt.setString(2, password);
      pstmt.setString(3, email);

      int result = pstmt.executeUpdate();
      success = (result > 0);

    } catch (SQLIntegrityConstraintViolationException e) {
      errorMessage = "이미 사용 중인 아이디 또는 이메일입니다.";
    } catch (Exception e) {
      e.printStackTrace();
      errorMessage = "서버 오류가 발생했습니다.";
    }

    // 🔽 이 위치 중요!
    if (success) {
    	  System.out.println("[✔] 회원가입 성공 → 로그인 페이지로 이동");
    	  response.sendRedirect("login.jsp");
    	} else {
    	  System.out.println("[✘] 회원가입 실패 → 다시 signup.jsp로 이동");
    	  request.setAttribute("error", errorMessage);
    	  RequestDispatcher rd = request.getRequestDispatcher("signup.jsp");
    	  rd.forward(request, response);
    	}

  }
}
