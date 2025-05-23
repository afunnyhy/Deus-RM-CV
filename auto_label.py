import json
import os


class RMArmorDataset:
    """
    用于自动检查使用labelme标注的RoboMaster装甲板数据集的合法性并自动分类点标签
    """

    def __init__(self, armor_branches=None, is_classify_points=True, files_path='.'):
        """
        初始化数据集检查和标签分类类
        :param armor_branches: 不同兵种矩形框标签的后缀，默认为RM3v3 ["sb", "1", "3"] (哨兵,英雄,步兵),传入 [""] 为不按照兵种分类
        :param is_classify_points: 是否对将点按颜色和方向分成8类，False为将点的标签全设置为point，默认为 True
        :param files_path: 数据集所在的文件夹路径，默认为当前目录
        """
        if armor_branches is None:  # 默认为RM3v3 ["sb", "1", "3"]
            armor_branches = ["sb", "1", "3"]  # [""] 为不按照兵种分类
        self.armor_branches = armor_branches  # 不同兵种矩形框标签的后缀
        if is_classify_points:  # 将点按颜色和方向分成8类
            self.point_labels = ["lub", "ldb", "rub", "rdb", "lur", "ldr", "rur", "rdr"]
        else:  # 将点的标签全设置为"point"
            self.point_labels = ["point"] * 8
        self.files_path = files_path  # 数据集文件夹路径
        self.count_json = 0  # json文件数量
        self.count_failed = 0  # 分类失败数量

    @staticmethod  # 四个点进行排序的静态方法
    def sort_points(points) -> list:
        """
        对四个点进行排序，以确保它们按正确的顺序排列
        :param points: 四个点组成的列表
        :return: 排序后的点
        """
        points.sort(key=lambda x: x[0])  # 按 x 坐标排序
        if points[0][1] > points[1][1]:
            points[0], points[1] = points[1], points[0]
        if points[2][1] > points[3][1]:
            points[2], points[3] = points[3], points[2]
        return points

    def classify_points(self, json_path) -> None:
        """
        对给定的 JSON 文件进行分类检查与点标签分类
        :param json_path: JSON 文件路径
        """
        # 将该文件重命名，文件名后加上“_temp”
        temp_json_path = json_path.replace(".json", "_temp.json")
        os.rename(json_path, temp_json_path)
        self.count_json += 1  # 统计数据集数量加1

        with open(temp_json_path, 'r', encoding='utf-8') as temp_file:
            data = json.load(temp_file)

        points_array = []
        red_rectangles = {branch: {} for branch in self.armor_branches}  # 为不同兵种红色矩形框的后缀创建字典
        blue_rectangles = {branch: {} for branch in self.armor_branches}  # 为不同兵种蓝色矩形框的后缀创建字典
        count_rectangles = 0
        new_shapes = []
        is_legal = True  # 数据集合法性标识

        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "point":  # 筛选 shape_type 为 point 的 shapes，并从原始数据中删除
                points = shape.get("points", [])
                points_array.append(points[0])
            else:
                new_shapes.append(shape)
                if shape.get("shape_type") == "rectangle":  # 筛选 shape_type 为 rectangle 的 shapes 并按颜色和兵种分类存储
                    count_rectangles += 1
                    # 获取矩形框的四个点并排序
                    points_temp = shape.get("points", [])
                    points = (min(points_temp[0][0], points_temp[1][0]), min(points_temp[0][1], points_temp[1][1]),
                              max(points_temp[0][0], points_temp[1][0]), max(points_temp[0][1], points_temp[1][1]))
                    if shape.get("label")[0:3] == "red":
                        armor_type = shape.get("label")[3:]
                        if armor_type in self.armor_branches:
                            red_rectangles[armor_type][points] = []
                        else:
                            print(f"\033[33m数据集不合法: 存在无效的矩形框标签{shape.get('label')}\033[0m")
                            is_legal = False
                    elif shape.get("label")[0:4] == "blue":
                        armor_type = shape.get("label")[4:]
                        if armor_type in self.armor_branches:
                            blue_rectangles[armor_type][points] = []
                        else:
                            print(f"\033[33m数据集不合法: 存在无效的矩形框标签{shape.get('label')}\033[0m")
                            is_legal = False
                    else:
                        print(f"\033[33m数据集不合法: 存在无效的矩形框标签{shape.get('label')}\033[0m")
                        is_legal = False

        left_points = []
        match_legal = True

        # 匹配点到矩形内
        for point in points_array:
            match_times = 0
            for red_rectangle_type in red_rectangles:
                for rectangle in red_rectangles[red_rectangle_type]:
                    if rectangle[0] <= point[0] <= rectangle[2] and rectangle[1] <= point[1] <= rectangle[3]:
                        red_rectangles[red_rectangle_type][rectangle].append(point)
                        match_times += 1
            for blue_rectangle_type in blue_rectangles:
                for rectangle in blue_rectangles[blue_rectangle_type]:
                    if rectangle[0] <= point[0] <= rectangle[2] and rectangle[1] <= point[1] <= rectangle[3]:
                        blue_rectangles[blue_rectangle_type][rectangle].append(point)
                        match_times += 1
            if match_times == 0:
                left_points.append(point)
            elif match_times > 1:
                match_legal = False
                break

        # 检查数据点合法性
        if count_rectangles == 0:
            print("\033[33m该图像可能被遗漏打标签: 数据集内未检测到任何矩形框\033[0m")
            is_legal = False
        if not match_legal:
            print("\033[33m数据集不合法: 有点同时在多个矩形框内\033[0m")
            is_legal = False
        if len(left_points) != 0:
            print("\033[33m数据集不合法: 有点不在矩形框内\033[0m")
            is_legal = False
        for red_rectangle_type in red_rectangles:
            for rectangle in red_rectangles[red_rectangle_type]:
                if len(red_rectangles[red_rectangle_type][rectangle]) != 4:
                    print(f"\033[33m数据集不合法: 存在标签为red{red_rectangle_type}的红色矩形框内点数为"
                          f"{len(red_rectangles[red_rectangle_type][rectangle])}不为4\033[0m")
                    is_legal = False
        for blue_rectangle_type in blue_rectangles:
            for rectangle in blue_rectangles[blue_rectangle_type]:
                if len(blue_rectangles[blue_rectangle_type][rectangle]) != 4:
                    print(f"\033[33m数据集不合法: 存在标签为blue{blue_rectangle_type}的蓝色矩形框内点数为"
                          f"{len(blue_rectangles[blue_rectangle_type][rectangle])}不为4\033[0m")
                    is_legal = False

        if not is_legal:  # 数据不合法，分类失败
            os.rename(temp_json_path, json_path)  # 改回文件名
            self.count_failed += 1  # 统计分类失败数量
            print(f"检查到错误标注(错误编号{self.count_failed})",
                  f"\033[31m-图像 {json_path} 数据集标注不合法(原因如上), 分类失败!\033[0m\n")
            return

        # 将点分成八个标签：lub,ldb,rub,rdb,lur,ldr,rur,rdr
        points_labels = [[], [], [], [], [], [], [], []]
        for blue_rectangle_type in blue_rectangles:
            for rectangle in blue_rectangles[blue_rectangle_type]:
                points = self.sort_points(blue_rectangles[blue_rectangle_type][rectangle])
                points_labels[0].append(points[0])  # lub
                points_labels[1].append(points[1])  # ldb
                points_labels[2].append(points[2])  # rub
                points_labels[3].append(points[3])  # rdb
        for red_rectangle_type in red_rectangles:
            for rectangle in red_rectangles[red_rectangle_type]:
                points = self.sort_points(red_rectangles[red_rectangle_type][rectangle])
                points_labels[4].append(points[0])  # lur
                points_labels[5].append(points[1])  # ldr
                points_labels[6].append(points[2])  # rur
                points_labels[7].append(points[3])  # rdr

        # 将点的标签添加到新的 shapes 中
        for i, label in enumerate(self.point_labels):
            for point in points_labels[i]:
                new_shapes.append({
                    "label": label,
                    "points": [point],
                    "group_id": None,
                    "shape_type": "point",
                    "flags": {}
                })

        # 更新原始 JSON 数据
        data["shapes"] = new_shapes

        # 将修改后的数据保存到输出文件
        with open(json_path, 'w', encoding='utf-8') as new_file:
            json.dump(data, new_file, ensure_ascii=False, indent=4)

        # 删除临时文件(原来旧的文件)
        os.remove(temp_json_path)

    def process_dataset(self) -> None:
        """
        批量处理指定路径下的所有 JSON 文件
        """
        files = os.listdir(self.files_path)
        print(f"\033[1m开始检查分类数据集,工作数据集目录:\033[0m",
              self.files_path if self.files_path != "." else "当前目录", end="\n\n")

        for file in files:
            if file.endswith(".json"):
                self.classify_points(os.path.join(self.files_path, file))

        if self.count_json == 0:
            print(f"\033[1;33m该目录下未找到任何数据集文件!请检查路径是否正确.\033[0m")
        elif self.count_failed == 0:
            print(self.count_json, "\033[1;32m份数据集未发现错误,全部分类成功.\033[0m")
        else:
            print(self.count_json, "\033[1;36m份数据集中有\033[0m", self.count_failed,
                  "\033[1;36m份数据集检查到\033[1;31m不合法\033[1;36m分类失败,其余分类成功.\033[0m")
        print(os.path.abspath(self.files_path), end=" ")  # 输出绝对路径
        input("\033[34m目录下数据集检查分类完成, 按回车键退出...\033[0m")


if __name__ == '__main__':
    # 创建数据集处理实例
    dataset = RMArmorDataset(is_classify_points=False,
                             files_path=r"E:\RoboMaster\装甲板数据集")  # 指定数据集文件夹路径,如r'.\dataset'
    dataset.process_dataset()  # 处理数据集
