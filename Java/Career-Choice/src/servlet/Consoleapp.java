package servlet;

import java.io.IOException;
import java.sql.SQLException;
import java.util.List;
import java.util.Scanner;

public class Consoleapp {
    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        while (true) {
            System.out.println("\n1.Create Table  2.Drop Table  3.Print Data");
            System.out.println("4.Update Cell   5.Delete Cell   6.Exit");
            System.out.print("Choose> ");

            int cmd;
            try {
                cmd = Integer.parseInt(scanner.nextLine().trim());
            } catch (NumberFormatException e) {
                System.out.println("Invalid input."); continue;
            }

            try {
                switch (cmd) {
                    case 1: DBManager.createTable();            break;
                    case 2: DBManager.dropTable();              break;
                    case 3: handlePrintData();                  break;
                    case 4: handleUpdateCell();                 break;
                    case 5: handleDeleteCell();                 break;
                    case 6: return;
                    default: System.out.println("Invalid choice.");
                }
            } catch (SQLException | IOException e) {
                System.err.println("Error: " + e.getMessage());
            }
        }
    }

    // 3) Print Data by year + industryType
    private static void handlePrintData() throws SQLException {
        String year = pickValue("year");
        String ind  = pickValue("industryType");
        DBManager.selectByIndustryYear(year, ind);
    }

    // 4) Update Cell
    private static void handleUpdateCell() throws SQLException {
        String year = pickValue("year");
        String ind  = pickValue("industryType");

        List<String> cols = DBManager.getColumnNames();
        // id,year,industryType는 수정 제외
        cols.remove("ID"); cols.remove("YEAR"); cols.remove("INDUSTRYTYPE");
        for (int i = 0; i < cols.size(); i++) {
            System.out.printf("%2d. %s%n", i+1, cols.get(i));
        }
        System.out.print("Select column> ");
        String col = cols.get(Integer.parseInt(scanner.nextLine()) - 1);

        System.out.print("New value> ");
        String val = scanner.nextLine();

        int cnt = DBManager.updateCell(year, ind, col, val);
        System.out.println("Updated rows: " + cnt);
    }

    // 5) Delete Cell
    private static void handleDeleteCell() throws SQLException {
        String year = pickValue("year");
        String ind  = pickValue("industryType");

        List<String> cols = DBManager.getColumnNames();
        cols.remove("ID"); cols.remove("YEAR"); cols.remove("INDUSTRYTYPE");
        for (int i = 0; i < cols.size(); i++) {
            System.out.printf("%2d. %s%n", i+1, cols.get(i));
        }
        System.out.print("Select column> ");
        String col = cols.get(Integer.parseInt(scanner.nextLine()) - 1);

        int cnt = DBManager.deleteCell(year, ind, col);
        System.out.println("Deleted cells (set NULL): " + cnt);
    }

    // year / industryType 목록을 보여주고 선택
    private static String pickValue(String column) throws SQLException {
        List<String> vals = DBManager.getDistinctValues(column);
        for (int i = 0; i < vals.size(); i++) {
            System.out.printf("%2d. %s%n", i+1, vals.get(i));
        }
        System.out.print("Pick " + column + "> ");
        return vals.get(Integer.parseInt(scanner.nextLine()) - 1);
    }
}