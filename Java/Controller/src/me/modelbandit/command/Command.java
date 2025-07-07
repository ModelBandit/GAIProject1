package me.modelbandit.command;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public interface Command {
	public void process(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException;
}
