#include<iostream>
using namespace std;

int main()
{
	double price;
	cout << "请输入商品价格(整数且不大于100元)：";
	cin >> price;
	if (price > 100 || price < 0 || (int)price != price)
	{
		cout << "商品价格不符合要求!" << endl;
		return 0;
	}
	double pay;
	cout << "请输入付款金额(整数且不大于100元)：";
	cin >> pay;
	if (pay > 100 || pay < 0 || (int)pay != pay)
	{
		cout << "付款金额不符合要求!" << endl;
		return 0;
	}
	if (pay < price)
	{
		cout << "付款金额必须大于商品价格！" << endl;
		return 0;
	}
	if (pay == price)
	{
		cout << "无需找零！" << endl;
		return 0;
	}
	int changes[6] = { 50, 20, 10, 5, 2, 1 };
	int rest = pay - price;
	cout << "最佳找零组合：" << endl;
	for (int i = 0; rest > 0 && i < 6; i++)
	{
		int cnt = 0;
		cnt = rest / changes[i];
		rest -= cnt * changes[i];
		if (cnt)
		{
			cout << changes[i] << "元：" << cnt << "张" << endl;
		}
	}
	return 0;
}