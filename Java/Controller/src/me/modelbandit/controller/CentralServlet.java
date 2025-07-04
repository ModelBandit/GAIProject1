package me.modelbandit.controller;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class CentralServlet
 */
@WebServlet("/CentralServlet")
public class CentralServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
    String projectRoot = "D:/myproject/GAIProject1/"; 
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public CentralServlet() {
        super();
//        System.out.println(projectRoot);
        String pyMainPath = "Python/main.py";
        
        String path = projectRoot+pyMainPath;
//        System.out.println(path);
        
        ProcessBuilder pb = new ProcessBuilder("python",path);
        
        Process proc;
        try {
			proc = pb.start();
			BufferedReader br = new BufferedReader(new InputStreamReader(proc.getInputStream()));
			String line;
			
			while((line = br.readLine()) != null) {
				System.out.println(line);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		response.getWriter().append("Served at: ").append(request.getContextPath());
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
		doGet(request, response);
	}

}
