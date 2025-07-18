package servlet;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;
import java.io.*;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;

// KlipsDataService는 데이터 로딩을 담당하는 클래스입니다.
class KlipsDataService {
    private static Map<Integer, Map<String, Double>> jobCategoryStats = new HashMap<>();
    private static boolean isLoaded = false;

    public static synchronized void loadKlipsData(String csvFilePath) {
        if (isLoaded) return;
        System.out.println("Attempting to load KLIPS data from: " + csvFilePath);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFilePath), StandardCharsets.UTF_8))) {
            br.readLine(); // 헤더 라인 스킵
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 4) continue;
                try {
                    int jobCategory = Integer.parseInt(parts[0].trim());
                    Map<String, Double> stats = new HashMap<>();
                    stats.put("job_category_income_avg", Double.parseDouble(parts[1].trim()));
                    stats.put("job_category_education_avg", Double.parseDouble(parts[2].trim()));
                    stats.put("job_category_satisfaction_avg", Double.parseDouble(parts[3].trim()));
                    jobCategoryStats.put(jobCategory, stats);
                } catch (NumberFormatException e) {
                    System.err.println("Skipping malformed data line in CSV: " + line);
                }
            }
            isLoaded = true;
            System.out.println("KLIPS data loaded successfully. Total categories: " + jobCategoryStats.size());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Map<String, Double> getJobCategoryStatistics(int jobCategory) {
        return jobCategoryStats.getOrDefault(jobCategory, Collections.emptyMap());
    }
}

// 사용자 입력을 캡슐화하는 간단한 데이터 클래스
class UserInput {
    double age, gender, education, monthlyIncome, jobSatisfaction, currentJobCategory;
    String satisFocusKey;
    
    public UserInput(HttpServletRequest request) {
        this.age = Double.parseDouble(request.getParameter("age"));
        this.gender = Double.parseDouble(request.getParameter("gender"));
        this.education = Double.parseDouble(request.getParameter("education"));
        this.monthlyIncome = Double.parseDouble(request.getParameter("monthly_income"));
        this.jobSatisfaction = Double.parseDouble(request.getParameter("job_satisfaction"));
        this.currentJobCategory = Double.parseDouble(request.getParameter("current_job_category"));
        this.satisFocusKey = request.getParameter("satis_focus_key");
    }
}

@WebServlet(urlPatterns = "/SimulateServlet", loadOnStartup = 1)
public class SimulateServlet extends HttpServlet {

    @Override
    public void init() throws ServletException {
        super.init();
        String klipsDataPath = getServletContext().getRealPath("/") + "/WEB-INF/data/klips_data_for_stats.csv";
        KlipsDataService.loadKlipsData(klipsDataPath);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setCharacterEncoding("UTF-8");

        try {
            UserInput userInput = new UserInput(request);
            JSONArray scenarios = buildScenarios(request, userInput);
            JSONArray predictionResults = runPythonPredict(scenarios);
            
            System.out.println("Prediction Results from Python: " + predictionResults.toString(4));
            
            processRequest(request, response, predictionResults);

        } catch (NumberFormatException e) {
            handleError(request, response, "입력값이 올바르지 않습니다. 숫자 형식에 맞춰 입력해주세요.", e);
        } catch (Exception e) {
            handleError(request, response, "예측 시스템 처리 중 오류가 발생했습니다.", e);
        }
    }

    private JSONArray buildScenarios(HttpServletRequest request, UserInput input) throws IOException {
        Map<String, Object> baseFeatures = createBaseFeatures(input);
        
        String jobACategoryParam = request.getParameter("job_A_category");
        String jobBCategoryParam = request.getParameter("job_B_category");
        
        double jobACategory = (jobACategoryParam != null) ? Double.parseDouble(jobACategoryParam) : 2;
        double jobBCategory = (jobBCategoryParam != null) ? Double.parseDouble(jobBCategoryParam) : 3;

        if (jobACategoryParam == null) { // 최초 요청 시에만 기본값 설정
             jobBCategory = 3;
        }

        JSONArray scenarios = new JSONArray();
        scenarios.put(createScenario(baseFeatures, input.currentJobCategory));
        scenarios.put(createScenario(baseFeatures, jobACategory));
        scenarios.put(createScenario(baseFeatures, jobBCategory));

        setAttributesForJSP(request, input, jobACategory, jobBCategory);
        return scenarios;
    }

    private Map<String, Object> createBaseFeatures(UserInput input) {
        Map<String, Object> features = new HashMap<>();
        features.put("age", input.age);
        features.put("gender", input.gender);
        features.put("education", input.education);
        features.put("monthly_income", input.monthlyIncome);
        features.put("job_satisfaction", input.jobSatisfaction);
        features.put("prev_job_satisfaction", input.jobSatisfaction);
        features.put("prev_monthly_income", input.monthlyIncome);

        final double WEIGHT_FACTOR = 1.2;
        List<String> satisfactionKeys = Arrays.asList("satis_wage", "satis_stability", "satis_task_content", "satis_work_env", "satis_work_time", "satis_growth", "satis_communication", "satis_fair_eval", "satis_welfare");
        
        for (String key : satisfactionKeys) {
            features.put(key, input.satisFocusKey.equals(key) ? input.jobSatisfaction * WEIGHT_FACTOR : input.jobSatisfaction);
        }
        return features;
    }

    private JSONObject createScenario(Map<String, Object> baseFeatures, double jobCategory) throws IOException {
        JSONObject scenario = new JSONObject(baseFeatures);
        scenario.put("job_category", jobCategory);
        
        addJobCategoryDerivedFeatures(scenario, (int) jobCategory, (double) baseFeatures.get("monthly_income"), (double) baseFeatures.get("education"));
        addInteractionFeatures(scenario, (double) baseFeatures.get("age"), (double) baseFeatures.get("monthly_income"), (double) baseFeatures.get("education"), jobCategory);
        
        return scenario;
    }

    private void addJobCategoryDerivedFeatures(JSONObject scenario, int jobCategory, double monthlyIncome, double education) {
        Map<String, Double> stats = KlipsDataService.getJobCategoryStatistics(jobCategory);
        double incomeAvg = stats.getOrDefault("job_category_income_avg", 0.0);
        double eduAvg = stats.getOrDefault("job_category_education_avg", 0.0);

        scenario.put("job_category_income_avg", incomeAvg);
        scenario.put("income_relative_to_job", (incomeAvg != 0) ? monthlyIncome / incomeAvg : 1.0);
        scenario.put("job_category_education_avg", eduAvg);
        scenario.put("education_relative_to_job", (eduAvg != 0) ? education / eduAvg : 1.0);
        scenario.put("job_category_satisfaction_avg", stats.getOrDefault("job_category_satisfaction_avg", 0.0));
    }

    private void addInteractionFeatures(JSONObject scenario, double age, double monthlyIncome, double education, double jobCategory) {
        scenario.put("age_x_job_category", age * jobCategory);
        scenario.put("monthly_income_x_job_category", monthlyIncome * jobCategory);
        scenario.put("education_x_job_category", education * jobCategory);
        scenario.put("income_relative_to_job_x_job_category", scenario.optDouble("income_relative_to_job", 1.0) * jobCategory);
    }

    private JSONArray runPythonPredict(JSONArray scenarios) throws IOException, InterruptedException {
        String pythonScriptPath = getServletContext().getRealPath("/") + "/WEB-INF/python_scripts/predict.py";
        ProcessBuilder pb = new ProcessBuilder("python", pythonScriptPath);
        Process process = pb.start();

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8))) {
            writer.write(scenarios.toString());
        }

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            StringBuilder errorOutput = new StringBuilder();
            try (BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = errorReader.readLine()) != null) {
                    errorOutput.append(line);
                }
            }
            throw new IOException("Python script execution failed with exit code " + exitCode + ": " + errorOutput.toString());
        }
        return new JSONArray(output.toString());
    }

    private void processRequest(HttpServletRequest request, HttpServletResponse response, JSONArray predictionResults) throws ServletException, IOException {
        request.setAttribute("predictionResults", predictionResults.toString());
        request.getRequestDispatcher("simulateResult.jsp").forward(request, response);
    }

    private void handleError(HttpServletRequest request, HttpServletResponse response, String userMessage, Exception e) throws ServletException, IOException {
        e.printStackTrace();
        request.setAttribute("errorMessage", userMessage);
        request.getRequestDispatcher("simulate.jsp").forward(request, response);
    }
    
    private void setAttributesForJSP(HttpServletRequest request, UserInput input, double jobACategory, double jobBCategory) {
        request.setAttribute("userAge", (int) input.age);
        request.setAttribute("userGender", (int) input.gender);
        request.setAttribute("userEducation", (int) input.education);
        request.setAttribute("userMonthlyIncome", (int) input.monthlyIncome);
        request.setAttribute("userJobSatisfaction", (int) input.jobSatisfaction);
        request.setAttribute("satisFocusKey", input.satisFocusKey);
        request.setAttribute("currentJobCategory", (int) input.currentJobCategory);
        request.setAttribute("selectedJobACategory", (int) jobACategory);
        request.setAttribute("selectedJobBCategory", (int) jobBCategory);
    }
}
