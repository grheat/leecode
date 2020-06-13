import java.util.function.Consumer;
import java.util.function.Supplier;

public class Child {
    public static void main(String[] args) {
        Supplier<String> i = () -> "Car";
        Consumer<String> c = x -> System.out.print(x.toLowerCase());
        Consumer<String> d = x -> System.out.print(x.toUpperCase());
        c.andThen(d).accept(i.get());
        System.out.println();
    }


}
