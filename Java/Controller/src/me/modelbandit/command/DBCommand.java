package me.modelbandit.command;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class DBServlet
 */
@WebServlet("/DB")
public class DBCommand implements Command {
	private static final long serialVersionUID = 1L;
       
	@Override
	public void process(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException{
		
	}
}
