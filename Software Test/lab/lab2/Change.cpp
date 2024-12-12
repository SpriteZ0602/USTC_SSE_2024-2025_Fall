#include<iostream>
using namespace std;

int main()
{
	double price;
	cout << "��������Ʒ�۸�(�����Ҳ�����100Ԫ)��";
	cin >> price;
	if (price > 100 || price < 0 || (int)price != price)
	{
		cout << "��Ʒ�۸񲻷���Ҫ��!" << endl;
		return 0;
	}
	double pay;
	cout << "�����븶����(�����Ҳ�����100Ԫ)��";
	cin >> pay;
	if (pay > 100 || pay < 0 || (int)pay != pay)
	{
		cout << "���������Ҫ��!" << endl;
		return 0;
	}
	if (pay < price)
	{
		cout << "��������������Ʒ�۸�" << endl;
		return 0;
	}
	if (pay == price)
	{
		cout << "�������㣡" << endl;
		return 0;
	}
	int changes[6] = { 50, 20, 10, 5, 2, 1 };
	int rest = pay - price;
	cout << "���������ϣ�" << endl;
	for (int i = 0; rest > 0 && i < 6; i++)
	{
		int cnt = 0;
		cnt = rest / changes[i];
		rest -= cnt * changes[i];
		if (cnt)
		{
			cout << changes[i] << "Ԫ��" << cnt << "��" << endl;
		}
	}
	return 0;
}