package servlet;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.sql.*;
import java.util.*;

public class DBManager {
    private static final String CSV_FILE   = "E:/DB/merged_data.csv";
    private static final String TABLE_NAME = "MERGED_STATS";
    private static final String SEQ_NAME   = TABLE_NAME + "_SEQ";

    private static final String URL      = "jdbc:oracle:thin:@localhost:1521/xe";
    private static final String USER     = "scott";
    private static final String PASSWORD = "tiger";

    // CSV로부터 테이블 생성 및 데이터 적재
    public static void createTable() throws SQLException, IOException {
        List<String> header;
        try (BufferedReader br = new BufferedReader(
                 new FileReader(CSV_FILE, StandardCharsets.UTF_8))) {
            String line = br.readLine();
            if (line.startsWith("\uFEFF")) line = line.substring(1);
            header = Arrays.asList(line.split(",", -1));
        }

        // 기존 TABLE/SEQ 삭제
        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate("DROP TABLE " + TABLE_NAME + " PURGE");
        } catch (SQLException ignore) {}
        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate("DROP SEQUENCE " + SEQ_NAME);
        } catch (SQLException ignore) {}

        // CREATE TABLE
        StringBuilder ddl = new StringBuilder("CREATE TABLE ")
            .append(TABLE_NAME).append(" (");
        for (int i = 0; i < header.size(); i++) {
            String col = header.get(i);
            if (i == 0)          ddl.append(col).append(" NUMBER PRIMARY KEY, ");
            else if ("year".equalsIgnoreCase(col))
                                ddl.append(col).append(" VARCHAR2(4), ");
            else                ddl.append(col).append(" VARCHAR2(255), ");
        }
        ddl.setLength(ddl.length() - 2);
        ddl.append(")");

        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate(ddl.toString());
            System.out.println("Created table: " + TABLE_NAME);
        }

        // CREATE SEQUENCE
        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate("CREATE SEQUENCE " + SEQ_NAME + " START WITH 1 INCREMENT BY 1");
            System.out.println("Created sequence: " + SEQ_NAME);
        }

        // 배치 INSERT
        String colsPart = String.join(",", header);
        StringBuilder valsPart = new StringBuilder(SEQ_NAME + ".NEXTVAL");
        for (int i = 1; i < header.size(); i++) {
            valsPart.append(",?");
        }
        String sql = "INSERT INTO " + TABLE_NAME +
                     " (" + colsPart + ") VALUES (" + valsPart + ")";

        try (Connection conn = getConn();
             PreparedStatement ps = conn.prepareStatement(sql);
             BufferedReader br = new BufferedReader(
                 new FileReader(CSV_FILE, StandardCharsets.UTF_8))) {

            br.readLine(); // skip header
            String line; int batch = 0;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",", -1);
                for (int i = 1; i < parts.length; i++) {
                    ps.setString(i, parts[i]);
                }
                ps.addBatch();
                if (++batch % 500 == 0) ps.executeBatch();
            }
            ps.executeBatch();
            System.out.println("Imported rows: " + batch);
        }
    }

    // 테이블 및 시퀀스 삭제
    public static void dropTable() throws SQLException {
        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate("DROP TABLE " + TABLE_NAME + " PURGE");
            System.out.println("Dropped table: " + TABLE_NAME);
        } catch (SQLException ignore) {}
        try (Connection conn = getConn(); Statement st = conn.createStatement()) {
            st.executeUpdate("DROP SEQUENCE " + SEQ_NAME);
            System.out.println("Dropped sequence: " + SEQ_NAME);
        } catch (SQLException ignore) {}
    }

    // 같은 industryType+year 의 데이터만 출력
    public static void selectByIndustryYear(String year, String industry) throws SQLException {
        String sql = "SELECT * FROM " + TABLE_NAME +
                     " WHERE year = ? AND industryType = ?";
        try (Connection conn = getConn();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, year);
            ps.setString(2, industry);
            try (ResultSet rs = ps.executeQuery()) {
                ResultSetMetaData md = rs.getMetaData();
                int cols = md.getColumnCount();
                for (int i = 1; i <= cols; i++) System.out.print(md.getColumnName(i) + "\t");
                System.out.println("\n" + "-".repeat(80));
                while (rs.next()) {
                    for (int i = 1; i <= cols; i++) {
                        System.out.print(rs.getString(i) + "\t");
                    }
                    System.out.println();
                }
            }
        }
    }

    // 셀 수정 (year+industryType 기준)
    public static int updateCell(String year, String industry, String column, String newVal)
            throws SQLException {
        String sql = "UPDATE " + TABLE_NAME +
                     " SET " + column + " = ? WHERE year = ? AND industryType = ?";
        try (Connection conn = getConn();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, newVal);
            ps.setString(2, year);
            ps.setString(3, industry);
            return ps.executeUpdate();
        }
    }

    // 셀 삭제(=NULL) (year+industryType 기준)
    public static int deleteCell(String year, String industry, String column)
            throws SQLException {
        String sql = "UPDATE " + TABLE_NAME +
                     " SET " + column + " = NULL WHERE year = ? AND industryType = ?";
        try (Connection conn = getConn();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, year);
            ps.setString(2, industry);
            return ps.executeUpdate();
        }
    }

    // 특정 컬럼에 대한 Distinct 값 조회 (year나 industryType용)
    public static List<String> getDistinctValues(String column) throws SQLException {
        String sql = "SELECT DISTINCT " + column + " FROM " + TABLE_NAME + " ORDER BY " + column;
        List<String> list = new ArrayList<>();
        try (Connection conn = getConn();
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) list.add(rs.getString(1));
        }
        return list;
    }

    // 컬럼 목록 조회
    public static List<String> getColumnNames() throws SQLException {
        String sql = "SELECT * FROM " + TABLE_NAME + " WHERE 1=0";
        List<String> cols = new ArrayList<>();
        try (Connection conn = getConn();
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            ResultSetMetaData md = rs.getMetaData();
            for (int i = 1; i <= md.getColumnCount(); i++) {
                cols.add(md.getColumnName(i));
            }
        }
        return cols;
    }

    private static Connection getConn() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
}