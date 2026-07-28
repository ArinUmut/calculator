from decimal import Decimal, InvalidOperation
class Usable_Calculator:
    OP = {"+", "-", "*", "/", "**", "//", "%"}

    def read_decimal(self,prompt):
        while True:
            value = input(prompt).strip().replace(",", ".")
            try:
                return Decimal(value)
            except (InvalidOperation, ValueError):
                print("girdiniz hatalı, tekrar dene oe")
    def get_number(self):
        a = self.read_decimal("1. sayı")
        b = self.read_decimal("2. sayı")
        return a, b
    def get_operations(self):
        while True:
            y = input("İşlem (+, -, *, /, **, //, %): ").strip()
            if y in self.OP:
                return y
            print("hatalı işlem tipi")

    def calc(self, a, y, b):
        if y in {"/", "//", "%"} and b == Decimal("0"):
            raise ZeroDivisionError("sıfıra bölme amk")
        match y:
            case "+":   return a + b
            case "-":   return a - b
            case "*":   return a * b
            case "/":   return a / b
            case "**":  return a ** b
            case "//":  return a // b
            case "%":   return a % b
            case _:     raise ValueError("düzgün işlem ver aq")
    
    def main(self):

        y = self.get_operations()
        a, b = self.get_number()

        try:
            z = self.calc(a, y, b)
            if z == z.to_integral_value():
                print(f"Sonuç:{int(z)} \n")
            else:
                print(f"Sonuç: {z}\n")
        except ZeroDivisionError as e:
            print(f"Hata: {e}\n")

if __name__ == "__main__":
    hesap_makinesi = Usable_Calculator()
    hesap_makinesi.main()
