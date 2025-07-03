
/*
#include <pdal/pdal.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/filters/PMFFilter.hpp>
#include <pdal/io/PCDReader.hpp>
#include <pdal/io/PCDWriter.hpp>
#include <pdal/PointView.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/Options.hpp>

int main()
{
	pdal::StageFactory factory;

	// 1. PCD Reader
	pdal::Stage* reader = factory.createStage("readers.pcd");
	pdal::Options readOpts;
	readOpts.add("filename", "C:/testpcl3/subsampled_removeOutliers_drone.pcd");
	reader->setOptions(readOpts);
	std::cout << "1" << std::endl;

	// 2. PMF Filter
	pdal::PMFFilter pmf;
	pdal::Options pmf_opts;
	pmf_opts.add("max_window_size", 33);
	pmf_opts.add("slope", 1.0);
	pmf_opts.add("initial_distance", 0.5);
	pmf_opts.add("max_distance", 2.5);
	pmf.setOptions(pmf_opts);
	pmf.setInput(*reader);
	std::cout << "2" << std::endl;


	// 3. PCD Writer
	pdal::Stage* writer = factory.createStage("writers.pcd");
	pdal::Options writeOpts;
	// writeOpts.add("filename", "C:/testpcl2/classified_pmf_allpoints.pcd");
	writeOpts.add("filename", "C:/testpcl3/classify_ground_points_label_drone.pcd");
	// writeOpts.add("extra_dims", "all"); // 기본 필드(x, y, z) 외의 추가 필드를 포함, Classification 필드 포함시키기 위함
	// writeOpts.add("keep_unspecified", true); // PDAL이 기본 필드 외 필드를 버리지 않게 함, 안 하면 Classification 필드 삭제됨
	// writeOpts.add("order", "X,Y,Z,label"); // 출력 순서 지정 및 Classification -> label로 이름 변경, PCL에서 읽히게 하기 위함
	writer->setOptions(writeOpts);
	writer->setInput(pmf);
	std::cout << "3" << std::endl;

	// 4. Execute pipeline
	pdal::PointTable table;
	writer->prepare(table);
	writer->execute(table);
	std::cout << "4" << std::endl;


	return 0;
}
*/


// PCL ProgressiveMorphologicalFilter segmentation
/*
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>


int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointIndicesPtr ground(new pcl::PointIndices);

	std::string input_pcd = "C:/testpcl3/subsampled_pocheon.pcd";
	std::string output_pcd = "C:/testpcl3/ground_points_pocheon.pcd";


	// Fill in the cloud data
	pcl::PCDReader reader;
	// Replace the path below with the path where you saved your file
	reader.read<pcl::PointXYZ>(input_pcd, *cloud);

	std::cerr << "Cloud before filtering: " << std::endl;
	//std::cerr << *cloud << std::endl;

	// Create the filtering object
	// PCL의 ProgressiveMorphologicalFilter (PMF) 필터를 사용하여 LiDAR 포인트 클라우드에서 지형(ground) 을 추출합니다.
	// ProgressiveMorphologicalFilter : 지면 필터링 알고리즘으로, 여러 창 크기(window size)를 점차 증가시키며 지면을 추정합니다.
	pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
	pmf.setInputCloud(cloud);// 필터링할 포인트 클라우드 입력

	// Morphological 연산에서 사용할 최대 커널 크기. 보통 더 큰 창은 더 부드러운 지면을 탐색
	pmf.setMaxWindowSize(12);
	// 경사도 허용값. 주변 포인트와의 높이 차가 이 비율 이상이면 지면이 아님
	pmf.setSlope(1.0f);

	
	pmf.setInitialDistance(0.5f); // 초기 거리 임계값.이보다 높이 차이가 크면 지면 아님
	pmf.setMaxDistance(3.0f); // 최대 높이차 허용. 이보다 큰 높이차는 무조건 비지면으로 판단
	pmf.extract(ground->indices); // 추출된 지면점들의 인덱스를 ground에 저장

	// Create the filtering object
	// ExtractIndices : 위에서 추출된 지면 인덱스만 골라내어 새로운 포인트 클라우드로 만듭니다.
	pcl::ExtractIndices<pcl::PointXYZ> extract;

	extract.setInputCloud(cloud); //  전체 포인트 클라우드
	extract.setIndices(ground); // 지면 인덱스 설정
	extract.filter(*cloud_filtered); // 지면점만 필터링해서 cloud_filtered에 저장
	// 즉, cloud_filtered는 지면점만 남은 결과입니다.

	std::cerr << "Ground cloud after filtering: " << std::endl;
	//std::cerr << *cloud_filtered << std::endl;

	// pcl::PCDWriter writer;
	// writer.write<pcl::PointXYZ>("samp11-utm_ground.pcd", *cloud_filtered, false);

	if (pcl::io::savePCDFileBinary(output_pcd, *cloud_filtered) == -1)
	{
		PCL_ERROR("❌ PCD 파일 저장 실패.\n");
		return -1;
	}


	// Extract non-ground returns
	//extract.setNegative(true);
	//extract.filter(*cloud_filtered);

	//std::cerr << "Object cloud after filtering: " << std::endl;
	//std::cerr << *cloud_filtered << std::endl;

	// writer.write<pcl::PointXYZ>("samp11-utm_object.pcd", *cloud_filtered, false);

	//if (pcl::io::savePCDFileBinary("C:/testpcl3/non_ground_drone.pcd", *cloud_filtered) == -1)
	//{
	//	PCL_ERROR("❌ PCD 파일 저장 실패.\n");
	//	return -1;
//	}



	return 0;
}*/



// Remove Outliers from PCD file using StatisticalOutlierRemoval
// 아래 코드 실행할 때, 식별자 pop_t 에러 떠서, 
// C:\PCL_1_15_0\3rdParty\FLANN\include\flann\algorithms\dist.h  이 경로 파일 522번 줄에  typedef unsigned long long pop_t;  코드 추가함// 202011348가 직접 추가한 코드
/*
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	std::string input_pcd = "C:/testpcl3/3_kdTree_simple_above_4m_meanshift_drone.pcd";
	std::string output_pcd = "C:/testpcl3/3_rmOutliers_kdTree_simple_above_4m_meanshift_drone.pcd";

	// Fill in the cloud data
	pcl::PCDReader reader;
	// Replace the path below with the path where you saved your file
	reader.read<pcl::PointXYZ>(input_pcd, *cloud);

	std::cerr << "Cloud before filtering: " << std::endl;
	std::cerr << *cloud << std::endl;

	// Create the filtering object
	// PCL(Point Cloud Library)에서 제공하는 통계 기반 이상치 제거 필터
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor; // 필터 객체 생성 (템플릿은 사용할 포인트 타입)
	sor.setInputCloud(cloud); // 필터의 입력 클라우드를 설정 (원본 cloud)

	// 각 포인트에 대해 가장 가까운 20개의 이웃을 고려해서 평균 거리 계산
	// -> 클러스터 내부 포인트는 평균 거리가 작고, 외부의 노이즈는 평균 거리가 큼
	sor.setMeanK(20);

	// 평균 거리 분포의 표준편차 × 계수(1.0) 를 임계값으로 설정
    // -> 평균 거리 > (전체 평균 + 1.0 * 표준편차) → 이상치로 간주됨
    // -> 이 값이 작을수록 더 많은 점이 제거됨, 클수록 더 관대해짐
	sor.setStddevMulThresh(1.0);

	// 필터링 수행 후 정상적인 점들만 cloud_filtered에 저장
	sor.filter(*cloud_filtered);

	// 거리들의 전체 분포에서 평균과 표준편차 계산

	// 정리: 작동 원리
	// 모든 포인트에 대해 K개 이웃까지의 평균 거리 계산
	// 평균 거리들의 전체 분포에서 평균과 표준편차 계산
	// 각 점이 전체 평균 + n * 표준편차보다 멀면 → 이상치로 간주
	// 이상치 제거 or 추출
	
	// 표준편차는 데이터가 평균으로부터 얼마나 퍼져 있는지를 수치로 나타내는 값
	// 데이터들이 평균값을 중심으로 얼마나 흩어져 있는지를 나타냄



	std::cerr << "Cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;

	//pcl::PCDWriter writer;
	//writer.write<pcl::PointXYZ>(output_pcd, *cloud_filtered, false); // 정상 points


	if (pcl::io::savePCDFileBinary(output_pcd, *cloud_filtered) == -1)
	{
		PCL_ERROR("❌ PCD 파일 저장 실패.\n");
		return -1;
	}


	sor.setNegative(true); // 이상치 points를 추출하기 위해 setNegative(true) 설정
	sor.filter(*cloud_filtered);
	//writer.write<pcl::PointXYZ>("C:/testpcl3/outliers_drone.pcd", *cloud_filtered, false);

	if (pcl::io::savePCDFileBinary("C:/testpcl3/outliers_drone123.pcd", *cloud_filtered) == -1)
	{
		PCL_ERROR("❌ PCD 파일 저장 실패.\n");
		return -1;
	}



	return (0);
}
*/


// SubSampling 10% 
/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/random_sample.h>
#include <iostream>

int main() {
    using PointT = pcl::PointXYZ;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr sampled_cloud(new pcl::PointCloud<PointT>);

    std::string input_pcd = "C:/testpcl3/chm_above_4m_non_ground_normalized.pcd";
    std::string output_pcd = "C:/testpcl3/2xsubsampled_above_4m_non_ground_normalized_drone.pcd";
    std::cout << "1" << std::endl;
    
    pcl::PCDReader reader_pcd;
	  reader_pcd.read(input_pcd, *cloud);


    //if (pcl::io::loadPCDFile<PointT>(input_pcd, *cloud) == -1) {
    //    std::cerr << "PCD 파일 로드 실패" << std::endl;
    //    return -1;
    //}

    std::cout << "2" << std::endl;

    
    pcl::RandomSample<PointT> sampler; // RandomSample 필터 객체 생성.
    sampler.setInputCloud(cloud); // 입력 포인트 클라우드 설정 (cloud는 원본 데이터)
    sampler.setSample(static_cast<int>(cloud->size() * 0.1)); // 10% 샘플링 설정
    sampler.filter(*sampled_cloud); // 실제로 필터링 수행 → 결과를 sampled_cloud에 저장.

	std::cout << "샘플링된 포인트 수: " << sampled_cloud->size() << std::endl;


    std::cout << "3" << std::endl;

    //pcl::io::savePCDFileASCII(output_pcd, *sampled_cloud);
    if (pcl::io::savePCDFileBinary(output_pcd, *sampled_cloud) == -1)
    {
        PCL_ERROR("❌ PCD 파일 저장 실패.\n");
        return -1;
    }
    std::cout << "Subsampled 파일 저장 완료: " << output_pcd << std::endl;
    std::cout << "4" << std::endl;

    return 0;
}
*/


// 파일 불러오기 
/*
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <iostream>


int main()
{

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::string output_pcd = "C:/testpcl3/pocheon_modified.pcd";

	pcl::PCDReader reader_pcd;
	reader_pcd.read(output_pcd, *cloud);

	std::cout << cloud->points.size() << " points loaded from " << output_pcd << std::endl;




	return 0;

		
}
*/

// 대역폭 반경에 있는 점만 고려하는 코드(실행 시간 짦음)
// 주변 이웃 좌표 구할 때, 주변 환경(hs) 뿐만 아니라, hT도 고려하는 코드
/*
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h> // 주변 반경 이웃 검출을 위한 KD-트리

// ────────────── 커널 함수들 ──────────────

// 식 (2) - 가우시안 커널 (수평)
double gs(double xA, double yA, double xi, double yi, double hs) {
    double dx = xA - xi, dy = yA - yi;
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq <= hs * hs)
        return std::exp(-0.5 * (dist_sq / (hs * hs)));
    return 0.0;
}

// 식 (4) - 마스크 조건
bool mask(double zA, double zi, double hT) {
    return (zA - hT / 4.0 <= zi) && (zi <= zA + hT / 2.0);
}

// 식 (5) - 비정규화 거리
double dist(double zA, double zi, double hT) {
    if (!mask(zA, zi, hT)) return 0.0;
    double denom = (3.0 * hT / 8.0);
    double t1 = std::abs((zA - hT / 4.0 - zi) / denom);
    double t2 = std::abs((zA + hT / 2.0 - zi) / denom);
    return std::min(t1, t2);
}

// 식 (3) - 방향 커널 (수직)
double gt(double zA, double zi, double hT) {
    if (!mask(zA, zi, hT)) return 0.0;
    double d = dist(zA, zi, hT);
    return 1.0 - (1.0 - d) * (1.0 - d); // 1.0 - d * d;
}

// 유클리드 거리
double shift_magnitude(const pcl::PointXYZ& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

*/
// 주변 반경 필터링 (hs와 hT 모두 고려), 수정 코드
// 3차원 필터: hs는 수평 반경, hT는 수직 반경
// 수작업으로 주변 이웃을 필터링하여, 수평 반경(hs)과 수직 반경(hT) 모두를 고려
/*
std::vector<pcl::PointXYZ> get_neighbors(const pcl::PointXYZ& center, const pcl::PointCloud<pcl::PointXYZ>& cloud, double hs, double hT) {
    std::vector<pcl::PointXYZ> neighbors;
    double hs_sq = hs * hs;
    double hT_half = hT / 2.0;

    for (const auto& pt : cloud.points) {
        double dx = center.x - pt.x;
        double dy = center.y - pt.y;
        double dz = center.z - pt.z;

        // 수평 반경 내 (x, y) 거리
        if ((dx * dx + dy * dy) <= hs_sq) {
            // 수직 높이 차이도 제한 내에 있으면 포함
            if (std::abs(dz) <= hT_half) {
                neighbors.push_back(pt);
            }
        }
    }
    return neighbors;
}
*/

// 주변 반경 필터링 (hs와 hT 모두 고려), KD-트리 기반
/*
std::vector<pcl::PointXYZ> get_neighbors_kdtree(const pcl::PointXYZ& center,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,  // 추가됨
    double hs, double hT)
{
    std::vector<pcl::PointXYZ> neighbors;


    // 반경 탐색 결과를 저장할 두 개의 리스트
    // point_indices: 반경 내 이웃 점들의 인덱스
    // point_squared_distances: 각 이웃과의 제곱 거리 (계산 효율을 위해 루트 없이 저장됨)
    std::vector<int> point_indices;
    // 변수 이름 해석
    // point_squared_distances	거리의 제곱 (d²)	✔️ 루트 안 씌운 값
    // point_distances	실제 거리(√d²)	✔️ 루트 씌운 값(보통 직접 계산 필요)
    std::vector<float> point_squared_distances;

    // 반경 기반 탐색 (hs는 수평 반경 → 3D 반경으로 사용 가능)
    // 3차원 거리 기준 반경 계산
    // 수평 반경 hs, 수직 반경 hT의 영향을 모두 반영한 원구 형태 거리
    // hT / 2는 위·아래 높이 범위를 고려한 것이고, 그 제곱이(hT²) / 4임

    // 3차원 거리의 일반 공식 :
    //    d = √(dx² + dy² + dz²)
    // 당신이 정의한 이웃 조건은 :
    //    dx² + dy² ≤ hs²
    //    | dz | ≤ hT / 2 → dz² ≤(hT / 2)² = hT² / 4

    // 👉 이 두 범위를 동시에 만족하는 최대 거리가 바로 :
    //    radius = √(hs² + (hT² / 4))
    //    즉,
    //        XY에서 최대 hs 떨어져 있고,
    //       Z축에서 최대 ±(hT / 2) 떨어져 있을 수 있으므로
    //    이 둘을 피타고라스 합한 3D 거리 최대값을 radius로 설정한 것입니다.


    // ✅ 예시

    //    hs = 1.0
    //    hT = 3.0
    //    radius = sqrt(1.0² + (3.0²) / 4.0) = sqrt(1.0 + 2.25) = sqrt(3.25) ≈ 1.802
    //    이 radius를 가지고 radiusSearch()를 하면 :

    // 반경 1.802m 안에 있는 점들을 찾되
    //    실제로는 나중에 다시 수평 / 수직 조건을 따로 걸러내는 구조입니다.


    double radius = std::sqrt(hs * hs + (hT * hT) / 4.0); // 3D 유클리드 반경

    // 반경 탐색 수행
    //    center를 중심으로 radius 거리 이내에 있는 점들의 인덱스를 반환
    //    반환된 점 수가 1개 이상일 경우에만 내부 for문 실행
    if (kdtree.radiusSearch(center, radius, point_indices, point_squared_distances) > 0) {
        for (int idx : point_indices) {

            // 인덱스로 해당 포인트 추출
            // cloud는 포인터이므로->points[idx]로 접근
            const auto& pt = cloud->points[idx];

            // 중심점과 이웃점 사이의 거리 계산 (X, Y, Z 방향별 차이)
            double dx = center.x - pt.x;
            double dy = center.y - pt.y;
            double dz = center.z - pt.z;

            // 수평 거리 체크 (x, y)
            // 수평 + 수직 거리 조건을 정확히 만족하는지 검사
            // dx² + dy² ≤ hs² → 수평 거리 반경 내
            // |dz| ≤ hT / 2 → 수직 높이 차이가 hT 범위 내

            // 이 조건은 radiusSearch()는 3D 거리 기준이기 때문에,
            // 수평 / 수직 분리된 조건을 다시 정제 필터링하는 단계입니다.
            if ((dx * dx + dy * dy) <= hs * hs) {
                // 수직 거리 체크 (z)
                if (std::abs(dz) <= hT / 2.0) {
                    neighbors.push_back(pt);
                }
            }
        }
    }

    return neighbors;
}



// Mean Shift 이동 벡터 계산 (필터링된 이웃만 사용)
pcl::PointXYZ compute_shift_vector(const pcl::PointXYZ& pA,
    const std::vector<pcl::PointXYZ>& neighbors,
    double hs, double hT)
{
    double num_x = 0, num_y = 0, num_z = 0, den = 0;

    for (const auto& pi : neighbors) {
        double weight = gs(pA.x, pA.y, pi.x, pi.y, hs) * gt(pA.z, pi.z, hT);
        num_x += pi.x * weight;
        num_y += pi.y * weight;
        num_z += pi.z * weight;
        den += weight;
    }

    pcl::PointXYZ result{ 0, 0, 0 };
    if (den > 1e-6) {
        result.x = (num_x / den) - pA.x;
        result.y = (num_y / den) - pA.y;
        result.z = (num_z / den) - pA.z;
    }
    return result;
}

// ────────────── main ──────────────
int main() {
    std::string input_path = "C:/testpcl3/2xsubsampled_above_4m_non_ground_normalized_drone.pcd";
    std::string output_path = "C:/testpcl3/2xsubsampled_kdTree_simple_above_4m_meanshift_drone.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *input) == -1) {
        std::cerr << "❌ 입력 파일 로드 실패\n";
        return -1;
    }

    std::cout << "▶ 포인트 수: " << input->points.size() << std::endl;



    // 복잡한 플롯(Complex plots)의 경우 :
    //    수평 대역폭(\(h ^ s\)) : 1.5 m
    //    수직 대역폭(\(h ^ r\)) : 5.0 m


    //  단순한 플롯(Simple plots)의 경우 :
    //     수평 대역폭(\(h ^ s\)) : 1.0 m
    //     수직 대역폭(\(h ^ r\)) : 3.5 m

    // 파라미터 설정
    double hs = 1.0;           // 수평 대역폭
    double hT = 3.5;           // 수직 대역폭
    int max_iter = 50;
    double disT = 1e-4;

    // PCL에서 제공하는 K-d Tree 구조체 생성
    // 빠른 거리 기반 탐색을 위한 공간 인덱싱 구조
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;


    // 탐색 대상이 되는 포인트 클라우드를 KdTree에 등록
    // 이후 radiusSearch() 등을 사용 가능하게 함
    kdtree.setInputCloud(input);

    for (const auto& pt : input->points) {
        pcl::PointXYZ current = pt;
        int iter = 0;



        // 주변 이웃 미리 필터링, HS와 HT 모두 고려, 수정 코드, 
        // std::vector<pcl::PointXYZ> neighbors = get_neighbors(current, *input, hs, hT);

        // 원래 함수 대신 사용
        // 주변 이웃 미리 필터링, KD-트리 기반
        // std::vector<pcl::PointXYZ> neighbors = get_neighbors_kdtree(current, input, hs, hT);
        std::vector<pcl::PointXYZ> neighbors = get_neighbors_kdtree(current, input, kdtree, hs, hT);


        while (true) {
            pcl::PointXYZ shift = compute_shift_vector(current, neighbors, hs, hT);
            double dis = shift_magnitude(shift);

            current.x += shift.x;
            current.y += shift.y;
            current.z += shift.z;

            iter++;
            if (dis < disT || iter >= max_iter) break;
        }

        output->points.push_back(current);
    }

    output->width = output->points.size();
    output->height = 1;
    output->is_dense = true;

    pcl::io::savePCDFileBinary(output_path, *output);
    std::cout << "✅ 최적화된 Mean Shift 완료: " << output_path << std::endl;

    return 0;
}
*/

// tree top, non-tree top을 각각 다른 pcd 파일로 저장하는 코드
// 이중 for문 ,시간 오래 걸림
/*
#include <iostream>
#include <vector>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

int main() {
    std::string input_path = "C:/testpcl3/4_kdTree_simple_above_4m_meanshift_drone.pcd";
    std::string tree_top_path = "C:/testpcl3/4_full_data_tree_top_only.pcd";
    std::string non_tree_top_path = "C:/testpcl3/4_full_data_non_tree_top_only.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tree_tops(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_tree_tops(new pcl::PointCloud<pcl::PointXYZRGB>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        std::cerr << "❌ 파일을 불러올 수 없습니다: " << input_path << std::endl;
        return -1;
    }

    std::cout << "▶ 총 포인트 수: " << cloud->points.size() << std::endl;

    double xy_radius = 3.0;
    double xy_radius_sq = xy_radius * xy_radius;
    int tree_top_count = 0;

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& pt = cloud->points[i];
        bool is_tree_top = true;

        for (size_t j = 0; j < cloud->points.size(); ++j) {
            if (i == j) continue;
            const auto& neighbor = cloud->points[j];

            double dx = pt.x - neighbor.x;
            double dy = pt.y - neighbor.y;
            double xy_dist_sq = dx * dx + dy * dy;

            if (xy_dist_sq <= xy_radius_sq && neighbor.z > pt.z) {
                is_tree_top = false;
                break;
            }
        }

        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt.x;
        pt_rgb.y = pt.y;
        pt_rgb.z = pt.z;

        if (is_tree_top) {
            pt_rgb.r = 255; pt_rgb.g = 0; pt_rgb.b = 0;
            tree_tops->points.push_back(pt_rgb);
            tree_top_count++;
        }
        else {
            pt_rgb.r = 128; pt_rgb.g = 128; pt_rgb.b = 128;
            non_tree_tops->points.push_back(pt_rgb);
        }
    }

    tree_tops->width = tree_tops->points.size();
    tree_tops->height = 1;
    tree_tops->is_dense = true;

    non_tree_tops->width = non_tree_tops->points.size();
    non_tree_tops->height = 1;
    non_tree_tops->is_dense = true;

    if (pcl::io::savePCDFileBinary(tree_top_path, *tree_tops) == -1) {
        std::cerr << "❌ Tree top 저장 실패: " << tree_top_path << std::endl;
        return -1;
    }

    if (pcl::io::savePCDFileBinary(non_tree_top_path, *non_tree_tops) == -1) {
        std::cerr << "❌ Non-tree top 저장 실패: " << non_tree_top_path << std::endl;
        return -1;
    }

    std::cout << "✅ Tree top 저장 완료: " << tree_top_path << std::endl;
    std::cout << "✅ Non-tree top 저장 완료: " << non_tree_top_path << std::endl;
    std::cout << "🌲 Tree top 수: " << tree_top_count << "개" << std::endl;

    return 0;
}
*/

// 평면 거리 (x,y) 기준 3m 이내에서 z값이 가장 큰 포인트만 tree top으로 간주하는 KD-Tree 최적화 코드
// (x,y ) 평면 반경 3m 내 and z 범위는 무한대 (수관 꼭대기 판별 반경) 내에 있는 점들 중에서 자기보다 z(높이)가 더 큰 점이 있는지 확인하여 수관 꼭대기를 판단하는 코드
// kd-tree 를 약간 변형하여 z값을 0으로 하여 z값 범위 무한대로 주는 것과 같은 효과 
/*
#include <iostream>
#include <vector>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

int main() {
    std::string input_path = "C:/testpcl3/4_kdTree_simple_above_4m_meanshift_drone.pcd";
    std::string tree_top_path = "C:/testpcl3/5_full_data_tree_top_only.pcd";
    std::string non_tree_top_path = "C:/testpcl3/5_full_data_non_tree_top_only.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        std::cerr << "❌ 파일을 불러올 수 없습니다: " << input_path << std::endl;
        return -1;
    }
    std::cout << "▶ 총 포인트 수: " << cloud->points.size() << std::endl;

    double radius = 3.0;
    int tree_top_count = 0;

    // KD-Tree 입력용 (z=0으로 평면 거리만 고려)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xy(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& pt : cloud->points)
        cloud_xy->push_back(pcl::PointXYZ(pt.x, pt.y, 0));

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_xy);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tree_top(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_tree_top(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& pt = cloud->points[i];
        pcl::PointXYZ query(pt.x, pt.y, 0);

        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(query, radius, indices, distances);

        bool is_tree_top = true;
        for (int idx : indices) {
            if (idx == i) continue;
            if (cloud->points[idx].z > pt.z) {
                is_tree_top = false;
                break;
            }
        }

        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = pt.x; pt_rgb.y = pt.y; pt_rgb.z = pt.z;

		std::cout << i << std::endl; // 디버깅용 출력
        if (is_tree_top) {
            pt_rgb.r = 255; pt_rgb.g = 0; pt_rgb.b = 0;
            tree_top->push_back(pt_rgb);
            tree_top_count++;
        }
        else {
            pt_rgb.r = 128; pt_rgb.g = 128; pt_rgb.b = 128;
            non_tree_top->push_back(pt_rgb);
        }
    }

    tree_top->width = tree_top->size(); tree_top->height = 1; tree_top->is_dense = true;
    non_tree_top->width = non_tree_top->size(); non_tree_top->height = 1; non_tree_top->is_dense = true;

    pcl::io::savePCDFileBinary(tree_top_path, *tree_top);
    pcl::io::savePCDFileBinary(non_tree_top_path, *non_tree_top);

    std::cout << "✅ Tree top 저장 완료: " << tree_top_path << " (" << tree_top_count << " points)\n";
    std::cout << "✅ Non-tree top 저장 완료: " << non_tree_top_path << " (" << non_tree_top->size() << " points)\n";

    return 0;
}
*/

// csv 파일 x,y,z 좌표로 저장하는 코드
#include <iostream>
#include <fstream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main() {
    std::string input_pcd = "C:/testpcl3/tree_top_only_XYZ.pcd";
    std::string output_csv = "C:/testpcl3/tree_top_only_XYZ.csv";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_pcd, *cloud) == -1) {
        std::cerr << "❌ PCD 파일 로드 실패: " << input_pcd << std::endl;
        return -1;
    }

    std::ofstream file(output_csv);
    if (!file.is_open()) {
        std::cerr << "❌ CSV 파일 저장 실패: " << output_csv << std::endl;
        return -1;
    }

    // 헤더
    file << "x,y,z\n";

    // 포인트 데이터 쓰기
    for (const auto& pt : cloud->points) {
        file << pt.x << "," << pt.y << "," << pt.z << "\n";
    }

    file.close();
    std::cout << "✅ CSV 저장 완료: " << output_csv << " (" << cloud->points.size() << " points)" << std::endl;

    return 0;
}







