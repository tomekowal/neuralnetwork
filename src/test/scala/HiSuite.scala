import org.scalatest.FunSuite    
    
import org.junit.runner.RunWith    
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner]) 
class HiSuite extends FunSuite { 

  test("one is really one") { 
    assert(1 === 1) 
  } 
}