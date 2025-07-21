package servlet;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CsvMerger {
    // CSV들이 들어 있는 폴더
    private static final Path CSV_DIR = Paths.get("E:/DB");
    // 병합된 CSV 출력 파일
    private static final Path OUT_FILE = CSV_DIR.resolve("merged_data.csv");
    // 원본 CSV 인코딩 (Windows-949 / CP949)
    private static final Charset SRC_CHARSET = StandardCharsets.UTF_8;

    // 병합 CSV 인코딩 (UTF-8)
    private static final Charset OUT_CHARSET = StandardCharsets.UTF_8;
    // 처리할 연도 범위
    private static final int START_YEAR = 2009;
    private static final int END_YEAR   = 2020;

    // 원본 컬럼(헤더) 배열
    private static final String[] COLUMNS = {
        "industryType","companyCount","ownerMaleRate","ownerFemaleRate",
        "singlePropCompanyRate","multiBusinessCompanyRate","U1D5CompanyRate",
        "U5D10CompanyRate","U10D20CompanyRate","U20D50CompanyRate",
        "U50D100CompanyRate","U100D300CompanyRate","U300CompanyRate",
        "workerCount","workerMaleRate","workerFemaleRate",
        "singlePropWorkerRate","multiBusinessWorkerRate","selfEmpFamilyWorkerRate",
        "fulltimeWorkerRate","dayWorkerRate","etcWorkerRate",
        "U1D5WorkerRate","U5D10WorkerRate","U10D20WorkerRate",
        "U20D50WorkerRate","U50D100WorkerRate","U100D300WorkerRate",
        "U300WorkerRate","avgAge","avgServYear","avgWorkDay",
        "avgTotalWorkTime","avgRegularWorkDay","avgOverWorkDay",
        "avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"
    };

}